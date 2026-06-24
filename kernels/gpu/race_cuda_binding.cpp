// PYBIND11 module exposing the causal RACE CUDA kernels.
//
// race_fused_fwd (forward_kernel.cu) and race_backward (backward_kernels.cu) are
// defined with external C++ linkage; torch.utils.cpp_extension.load() compiles
// the two .cu files together with this .cpp into one module and the linker
// resolves the symbols (mirrors kernels/cpu/race_pref.cpp).
#include <torch/extension.h>
#include <vector>

using at::Tensor;

Tensor race_fused_fwd(Tensor probsK, Tensor probsQ, Tensor V2, float eps, int64_t chunk);
std::vector<Tensor> race_backward(Tensor probsK, Tensor probsQ, Tensor V2, Tensor grad_out,
                                  float eps, int64_t chunk);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("race_fused_fwd", &race_fused_fwd,
          "Causal RACE forward (chunked parallel scan): "
          "(probsK[N,T,S], probsQ[N,T,S], V2[N,T,D], eps, chunk) -> out[N,T,D]");
    m.def("race_backward", &race_backward,
          "Causal RACE exact backward (forward-scan, chunked): "
          "(probsK, probsQ, V2, grad_out, eps, chunk) -> (gradProbsK, gradProbsQ, gradV)");
}
