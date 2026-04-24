// elsa_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float<c10::Half>(c10::Half x) {
#if __CUDA_ARCH__ >= 530
    __half h = *reinterpret_cast<const __half*>(&x);
    return __half2float(h);
#else
    return static_cast<float>(x);
#endif
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ double from_float<double>(float x) {
    return static_cast<double>(x);
}

template <>
__device__ __forceinline__ c10::Half from_float<c10::Half>(float x) {
#if __CUDA_ARCH__ >= 530
    __half h = __float2half_rn(x);
    return *reinterpret_cast<c10::Half*>(&h);
#else
    return c10::Half(x);
#endif
}

// 前向：每個 thread 處理一個 query row (i)
// BLOCK_M: 每個 block 一次處理的 rows 數
template <typename scalar_t, bool CAUSAL, int BLOCK_M = 128, int MAX_D = 128>
__global__ void elsa_forward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O,
    // shape
    int64_t B, int64_t H, int64_t N, int64_t D,
    // strides
    int64_t s_qb, int64_t s_qh, int64_t s_qn, int64_t s_qd,
    int64_t s_kb, int64_t s_kh, int64_t s_kn, int64_t s_kd,
    int64_t s_vb, int64_t s_vh, int64_t s_vn, int64_t s_vd,
    int64_t s_ob, int64_t s_oh, int64_t s_on, int64_t s_od,
    // scale
    float scale
){
    const int64_t bh = blockIdx.x;                 // [0, B*H)
    const int64_t b  = bh / H;
    const int64_t h  = bh % H;

    const int64_t i0 = blockIdx.y * blockDim.x + threadIdx.x;  // row start
    if (i0 >= N) return;

    // 限制 head dim：MAX_D 覆蓋 64/80/96/128 常見 head_dim
    if (D > MAX_D) return; // 防呆：避免 VLA 或 local 溢位（要更大可把 MAX_D 調高）

    // 指到 (b,h,i0,0)
    const scalar_t* q_row = Q + b*s_qb + h*s_qh + i0*s_qn;

    // 線上 softmax 累積器
    float m = -INFINITY;          // running max
    float l = 0.f;                 // running sum
    float acc[MAX_D];              // running weighted sum
#pragma unroll
    for (int d = 0; d < MAX_D; ++d) acc[d] = 0.f;

    // 掃過所有 key/value
    for (int64_t j = 0; j < N; ++j) {
        if (CAUSAL && j > i0) break;

        // s = <q_i, k_j> * scale
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < MAX_D; ++d) {
            if (d >= D) break;
            const float qv = to_float<scalar_t>(q_row + d*s_qd >= Q ? *(q_row + d*s_qd) : scalar_t(0));
            const scalar_t* k_row = K + b*s_kb + h*s_kh + j*s_kn;
            const float kv = to_float<scalar_t>(*(k_row + d*s_kd));
            s += qv * kv;
        }
        s *= scale;

        if (s > m) {
            const float beta = expf(m - s); // <= 1
            l = l * beta + 1.f;
            // acc = acc * beta + v_j
#pragma unroll
            for (int d = 0; d < MAX_D; ++d) {
                if (d >= D) break;
                const scalar_t* v_row = V + b*s_vb + h*s_vh + j*s_vn;
                const float vv = to_float<scalar_t>(*(v_row + d*s_vd));
                acc[d] = acc[d] * beta + vv;
            }
            m = s;
        } else {
            const float alpha = expf(s - m); // <= 1
            l += alpha;
#pragma unroll
            for (int d = 0; d < MAX_D; ++d) {
                if (d >= D) break;
                const scalar_t* v_row = V + b*s_vb + h*s_vh + j*s_vn;
                const float vv = to_float<scalar_t>(*(v_row + d*s_vd));
                acc[d] += alpha * vv;
            }
        }
    }

    // 正規化，寫回
    float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    scalar_t* o_row = O + b*s_ob + h*s_oh + i0*s_on;
#pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
        if (d >= D) break;
        const float outv = acc[d] * inv_l;
        *(o_row + d*s_od) = from_float<scalar_t>(outv);
    }
}

// launcher
template <typename scalar_t>
void elsa_forward_launch(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    bool causal, float scale)
{
    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t N = q.size(2);
    const int64_t D = q.size(3);

    dim3 block(128);
    dim3 grid(B*H, (N + block.x - 1) / block.x);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (causal) {
        elsa_forward_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale
        );
    } else {
        elsa_forward_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale
        );
    }
}

// Explicit instantiations to satisfy separate compilation/linking.
template void elsa_forward_launch<float>(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    bool causal, float scale);
template void elsa_forward_launch<double>(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    bool causal, float scale);
template void elsa_forward_launch<c10::Half>(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    bool causal, float scale);
