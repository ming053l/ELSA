#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

template <typename scalar_t>
__global__ void paper_scan_d64_final_kernel(
    const float* __restrict__ m,
    const float* __restrict__ z,
    const scalar_t* __restrict__ s,
    scalar_t* __restrict__ out,
    float* __restrict__ row_m,
    float* __restrict__ row_z,
    int q_start,
    int n_ctx,
    int k_blocks,
    int q_blocks,
    int bh,
    int block_m,
    bool store_state) {
  const int packed = blockIdx.x;
  const int row = packed % block_m;
  const int qb = packed / block_m;
  const int bh_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int q_idx = q_start + qb * block_m + row;
  if (q_idx >= n_ctx) {
    return;
  }

  extern __shared__ float shared[];
  float* reduce_sh = shared;
  float* weight_sh = reduce_sh + blockDim.x;
  float* partial_s_sh = weight_sh + k_blocks;

  float local_m = -CUDART_INF_F;
  for (int kb = tid; kb < k_blocks; kb += blockDim.x) {
    const int base = ((kb * q_blocks + qb) * bh + bh_idx) * block_m + row;
    local_m = fmaxf(local_m, m[base]);
  }
  reduce_sh[tid] = local_m;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_sh[tid] = fmaxf(reduce_sh[tid], reduce_sh[tid + stride]);
    }
    __syncthreads();
  }
  const float final_m = reduce_sh[0];

  float local_z = 0.0f;
  for (int kb = tid; kb < k_blocks; kb += blockDim.x) {
    const int base = ((kb * q_blocks + qb) * bh + bh_idx) * block_m + row;
    const float z_val = z[base];
    const float weight = z_val > 0.0f ? exp2f(m[base] - final_m) : 0.0f;
    weight_sh[kb] = weight;
    local_z += z_val * weight;
  }
  reduce_sh[tid] = local_z;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_sh[tid] += reduce_sh[tid + stride];
    }
    __syncthreads();
  }
  const float final_z = reduce_sh[0];

  if (store_state && tid == 0) {
    row_m[static_cast<int64_t>(bh_idx) * n_ctx + q_idx] = final_z > 0.0f ? final_m : -CUDART_INF_F;
    row_z[static_cast<int64_t>(bh_idx) * n_ctx + q_idx] = final_z;
  }

  const int lanes_per_d = blockDim.x >> 6;
  const int d = tid & 63;
  const int lane = tid >> 6;
  float local_s = 0.0f;
  for (int kb = lane; kb < k_blocks; kb += lanes_per_d) {
    const int base = ((kb * q_blocks + qb) * bh + bh_idx) * block_m + row;
    local_s += static_cast<float>(s[static_cast<int64_t>(base) * 64 + d]) * weight_sh[kb];
  }
  partial_s_sh[d * lanes_per_d + lane] = local_s;
  __syncthreads();

  if (lane == 0) {
    float s_sum = 0.0f;
    for (int idx = 0; idx < lanes_per_d; ++idx) {
      s_sum += partial_s_sh[d * lanes_per_d + idx];
    }
    const float value = final_z > 0.0f ? s_sum / final_z : 0.0f;
    out[(static_cast<int64_t>(bh_idx) * n_ctx + q_idx) * 64 + d] = static_cast<scalar_t>(value);
  }
}

void paper_scan_d64_final_cuda_launch(
    torch::Tensor m,
    torch::Tensor z,
    torch::Tensor s,
    torch::Tensor out,
    torch::Tensor row_m,
    torch::Tensor row_z,
    int64_t q_start,
    int64_t n_ctx,
    int64_t k_blocks,
    int64_t q_blocks,
    int64_t bh,
    int64_t block_m,
    bool store_state) {
  const at::cuda::CUDAGuard device_guard(s.device());
  const dim3 grid(
      static_cast<unsigned int>(q_blocks * block_m),
      static_cast<unsigned int>(bh));
  const int threads = 256;
  const size_t shared_bytes =
      static_cast<size_t>(threads + k_blocks + threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(s.scalar_type(), "paper_scan_d64_final_cuda", [&] {
    paper_scan_d64_final_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
        m.data_ptr<float>(),
        z.data_ptr<float>(),
        s.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        row_m.data_ptr<float>(),
        row_z.data_ptr<float>(),
        static_cast<int>(q_start),
        static_cast<int>(n_ctx),
        static_cast<int>(k_blocks),
        static_cast<int>(q_blocks),
        static_cast<int>(bh),
        static_cast<int>(block_m),
        store_state);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
