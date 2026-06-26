// PYBIND11 module exposing the fused single-token RACE decode CUDA kernel.
//
// race_decode_step (decode_kernel.cu) is defined with external C++ linkage;
// torch.utils.cpp_extension.load() compiles decode_kernel.cu together with this
// .cpp into one module and the linker resolves the symbol. This is a STANDALONE
// extension, fully separate from race_cuda (the training forward/backward).
#include <torch/extension.h>

using at::Tensor;

void race_decode_step(Tensor q, Tensor k, Tensor v, Tensor planes_T, Tensor protos_T,
                      Tensor A, Tensor B, Tensor out, double scale, double eps,
                      int64_t L, int64_t Kbits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("race_decode_step", &race_decode_step,
          "Fused single-token RACE decode (soft-hash + state update + readout): "
          "(q,k,v[N,hd], planes_T[hd,L*Kbits], protos_T[Kbits,R], A[N,S], B[N,S,hd], "
          "out[N,hd], scale, eps, L, Kbits) -> out; A,B updated in place");
}
