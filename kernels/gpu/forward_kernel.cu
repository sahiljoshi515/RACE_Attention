// ============================================================================
// Causal RACE forward — chunked parallel scan.
//
//   out[n,t,d] = sum_s probsQ[n,t,s] * B(t)[n,s,d] / (A(t)[n,s] + eps)
//   A(t)[n,s]   = sum_{tau<=t} probsK[n,tau,s]
//   B(t)[n,s,d] = sum_{tau<=t} probsK[n,tau,s] * V2[n,tau,d]
//
// The time axis is split into chunks of size C (G = ceil(T/C)). The scan runs in
// three phases so the GPU sees G*S*N-way parallelism and each block's serial loop
// is C rather than T:
//   phase1: per-chunk partial sums  cA[n,s,g]=sum_{t in g} pk,  cB[n,s,g,d]=sum pk*v
//   phase2: in-place EXCLUSIVE scan of cA/cB over g  -> per-chunk entry offsets
//   phase3: per-chunk readout from the offset (A+=pk; B+=pk*v; atomicAdd out)
//
// Numerically matches the PyTorch cumsum reference to fp32 precision. Used by the
// exact backward in backward_kernels.cu.
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

using at::Tensor;

// ---- phase1: per-chunk partial sums. grid=(G,S,N), block = padded D ----
__global__ void racefwd_phase1(
    const float *__restrict__ probsK, // [N,T,S]
    const float *__restrict__ V2,     // [N,T,D]
    float *__restrict__ cA,           // [N,S,G]
    float *__restrict__ cB,           // [N,S,G,D]
    int N, int T, int S, int D, int C, int G)
{
    int g = blockIdx.x, s = blockIdx.y, n = blockIdx.z, d = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T);
    if (t0 >= T) return;

    float lb = 0.0f, la = 0.0f;
    bool has_d = (d < D);
    for (int t = t0; t < t1; ++t)
    {
        float pk = probsK[((size_t)n * T + t) * S + s];   // broadcast across threads
        la += pk;
        if (has_d) lb += pk * V2[((size_t)n * T + t) * (size_t)D + d];
    }
    if (has_d) cB[(((size_t)n * S + s) * G + g) * (size_t)D + d] = lb;
    if (d == 0) cA[((size_t)n * S + s) * G + g] = la;
}

// ---- phase2: in-place EXCLUSIVE prefix scan over chunks g. grid=(S,N), block 256 ----
__global__ void racefwd_phase2(
    float *__restrict__ cA, float *__restrict__ cB, int N, int S, int D, int G)
{
    int s = blockIdx.x, n = blockIdx.y, tid = threadIdx.x;
    if (tid == 0)
    {
        float run = 0.0f;
        for (int g = 0; g < G; ++g)
        {
            size_t i = ((size_t)n * S + s) * G + g;
            float v = cA[i]; cA[i] = run; run += v;
        }
    }
    for (int d = tid; d < D; d += blockDim.x)
    {
        float run = 0.0f;
        for (int g = 0; g < G; ++g)
        {
            size_t i = (((size_t)n * S + s) * G + g) * (size_t)D + d;
            float v = cB[i]; cB[i] = run; run += v;
        }
    }
}

// ---- phase3: per-chunk readout from the offset. grid=(G,S,N), block = padded D ----
__global__ void racefwd_phase3(
    const float *__restrict__ probsK, // [N,T,S]
    const float *__restrict__ probsQ, // [N,T,S]
    const float *__restrict__ V2,     // [N,T,D]
    const float *__restrict__ cAoff,  // [N,S,G]
    const float *__restrict__ cBoff,  // [N,S,G,D]
    float *__restrict__ out,          // [N,T,D] (zeroed)
    int N, int T, int S, int D, int C, int G, float eps)
{
    int g = blockIdx.x, s = blockIdx.y, n = blockIdx.z, d = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T);
    if (t0 >= T || d >= D) return;

    float A = cAoff[((size_t)n * S + s) * G + g];
    float B = cBoff[(((size_t)n * S + s) * G + g) * (size_t)D + d];
    for (int t = t0; t < t1; ++t)
    {
        float pk = probsK[((size_t)n * T + t) * S + s];
        float pq = probsQ[((size_t)n * T + t) * S + s];
        size_t idxVD = ((size_t)n * T + t) * (size_t)D + d;
        A += pk;
        B += pk * V2[idxVD];
        atomicAdd(&out[idxVD], pq * (B / (A + eps)));
    }
}

static inline int pad_to_warp(int D)
{
    int b = ((D + 31) / 32) * 32;
    if (b > 1024) b = 1024;
    if (b < 32) b = 32;
    return b;
}

// ---- host wrapper: out = causal RACE forward (chunk size arg; <=0 -> default 8192) ----
Tensor race_fused_fwd(Tensor probsK, Tensor probsQ, Tensor V2, float eps, int64_t chunk)
{
    TORCH_CHECK(probsK.is_cuda() && probsQ.is_cuda() && V2.is_cuda(), "CUDA only");
    TORCH_CHECK(probsK.scalar_type() == at::kFloat && probsQ.scalar_type() == at::kFloat && V2.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(probsK.dim() == 3 && probsQ.dim() == 3 && V2.dim() == 3, "shapes [N,T,S],[N,T,S],[N,T,D]");
    int N = probsK.size(0), T = probsK.size(1), S = probsK.size(2), D = V2.size(2);
    int C = (int)chunk;
    if (C <= 0) C = 8192;
    if (C > T) C = T;
    int G = (T + C - 1) / C;

    auto fopt = torch::TensorOptions().device(probsK.device()).dtype(at::kFloat);
    auto out = torch::zeros({N, T, D}, fopt);
    auto cA = torch::empty({N, S, G}, fopt);
    auto cB = torch::empty({N, S, G, D}, fopt);

    int blk = pad_to_warp(D);
    dim3 g1((unsigned)G, (unsigned)S, (unsigned)N);
    racefwd_phase1<<<g1, blk>>>(probsK.data_ptr<float>(), V2.data_ptr<float>(),
                                cA.data_ptr<float>(), cB.data_ptr<float>(), N, T, S, D, C, G);
    dim3 g2((unsigned)S, (unsigned)N, 1);
    racefwd_phase2<<<g2, 256>>>(cA.data_ptr<float>(), cB.data_ptr<float>(), N, S, D, G);
    dim3 g3((unsigned)G, (unsigned)S, (unsigned)N);
    racefwd_phase3<<<g3, blk>>>(probsK.data_ptr<float>(), probsQ.data_ptr<float>(), V2.data_ptr<float>(),
                                cA.data_ptr<float>(), cB.data_ptr<float>(), out.data_ptr<float>(),
                                N, T, S, D, C, G, eps);
    return out;
}
