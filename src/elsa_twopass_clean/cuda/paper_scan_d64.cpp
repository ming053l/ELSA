#include <torch/extension.h>

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
    bool store_state);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("final_reduce", &paper_scan_d64_final_cuda_launch, "ELSA D64 paper-scan final reduction");
}
