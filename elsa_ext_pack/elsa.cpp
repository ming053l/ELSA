// elsa.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
void elsa_forward_launch(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    bool causal, float scale);

torch::Tensor elsa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           bool causal, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q/k/v dtypes must match");
    TORCH_CHECK(q.dim()==4 && k.sizes()==q.sizes() && v.sizes()==q.sizes(),
                "q/k/v must be (B,H,N,D) same shape");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto out = at::empty_like(q);

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "elsa_forward", [&]{
        elsa_forward_launch<scalar_t>(q_c, k_c, v_c, out, causal, static_cast<float>(scale));
    });

    // 立刻檢查 kernel 狀態，避免延後到你 access attn 才爆
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elsa_forward, "ELSA forward (CUDA)");
}
