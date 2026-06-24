// ============================================================================
// Causal RACE backward — exact forward-scan + chunked parallel scan.
//
// Computes gradients of the causal RACE forward (forward_kernel.cu) w.r.t.
// probsK, probsQ and V2. The causal prefix states A(t), B(t) are rebuilt by a
// FORWARD scan (never by reverse subtraction from finals, and never stored in
// fp16) so the gradients are numerically exact at all sequence lengths. Every
// scan is chunked over the time axis (chunk C, G=ceil(T/C)) for SM occupancy.
//
// Let G(t)[s] = sum_d grad_out[t,d]*B(t)[s,d], inv=1/(A(t)[s]+eps).
//   gradProbsQ[t,s] = G(t)[s]*inv
//   gradA(t)[s]     = -probsQ[t,s]*G(t)[s]*inv^2
//   gB(t)[s,d]      = probsQ[t,s]*grad_out[t,d]*inv
//   gAn(t)[s]       = sum_{tau>=t} gradA(tau)[s]   (suffix sum)
//   gBn(t)[s,d]     = sum_{tau>=t} gB(tau)[s,d]    (suffix sum)
//   gradProbsK[t,s] = gAn(t)[s] + sum_d gBn(t)[s,d]*V2[t,d]
//   gradV[t,d]      = sum_s probsK[t,s]*gBn(t)[s,d]
//
// pass1 (forward): chunk B/A offsets -> readout gradProbsQ, gradA, A_all.
// pass2 (reverse): per-chunk reverse totals of gB, gradA -> suffix offsets (shared
//                  by the gradProbsK and gradV readouts).
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>
#include <vector>

using at::Tensor;

__inline__ __device__ float warp_reduce_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__inline__ __device__ float block_reduce_sum(float v)
{
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}
static inline int pad_to_warp(int D) { int b = ((D + 31) / 32) * 32; return b < 32 ? 32 : (b > 1024 ? 1024 : b); }

// ---- pass1 chunk partial sums (fwd): cB[n,s,g,d]=sum pk*v, cA[n,s,g]=sum pk ----
__global__ void racebwd_p1_totals(const float *__restrict__ pK, const float *__restrict__ V2,
                                  float *__restrict__ cB, float *__restrict__ cA,
                                  int N, int T, int S, int D, int C, int G)
{
    int g = blockIdx.x, s = blockIdx.y, n = blockIdx.z, d = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T); if (t0 >= T) return;
    float lb = 0.f, la = 0.f; bool hd = d < D;
    for (int t = t0; t < t1; ++t) { float pk = pK[((size_t)n*T+t)*S+s]; la += pk; if (hd) lb += pk*V2[((size_t)n*T+t)*(size_t)D+d]; }
    if (hd) cB[(((size_t)n*S+s)*G+g)*(size_t)D+d] = lb;
    if (d == 0) cA[((size_t)n*S+s)*G+g] = la;
}

// ---- forward exclusive scan over g (in place): off[g] = sum_{g'<g} c[g'] ----
__global__ void racebwd_scan_fwd(float *__restrict__ cB, float *__restrict__ cA, int N, int S, int D, int G)
{
    int s = blockIdx.x, n = blockIdx.y, tid = threadIdx.x;
    if (tid == 0) { float r = 0.f; for (int g = 0; g < G; ++g) { size_t i = ((size_t)n*S+s)*G+g; float v = cA[i]; cA[i] = r; r += v; } }
    for (int d = tid; d < D; d += blockDim.x) { float r = 0.f; for (int g = 0; g < G; ++g) { size_t i = (((size_t)n*S+s)*G+g)*(size_t)D+d; float v = cB[i]; cB[i] = r; r += v; } }
}

// ---- pass1 readout (fwd): gradProbsQ, gradA, A_all ----
__global__ void racebwd_p1_readout(const float *__restrict__ pK, const float *__restrict__ pQ,
                                   const float *__restrict__ V2, const float *__restrict__ GO,
                                   const float *__restrict__ offA, const float *__restrict__ offB,
                                   float *__restrict__ gradProbsQ, float *__restrict__ gradA, float *__restrict__ A_all,
                                   int N, int T, int S, int D, int C, int G, float eps)
{
    int g = blockIdx.x, s = blockIdx.y, n = blockIdx.z, tid = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T); if (t0 >= T) return;
    extern __shared__ float shB[];               // [D]
    __shared__ float shA;
    for (int d = tid; d < D; d += blockDim.x) shB[d] = offB[(((size_t)n*S+s)*G+g)*(size_t)D+d];
    if (tid == 0) shA = offA[((size_t)n*S+s)*G+g];
    __syncthreads();
    for (int t = t0; t < t1; ++t)
    {
        size_t iKS = ((size_t)n*T+t)*S+s; float pk = pK[iKS], pq = pQ[iKS];
        if (tid == 0) shA += pk;
        __syncthreads();
        float A = shA; size_t bVD = ((size_t)n*T+t)*(size_t)D;
        float Gp = 0.f;
        for (int d = tid; d < D; d += blockDim.x) { float b = shB[d] + pk*V2[bVD+d]; shB[d] = b; Gp += GO[bVD+d]*b; }
        float Gr = block_reduce_sum(Gp);
        if (tid == 0) { float inv = 1.f/(A+eps); gradProbsQ[iKS] = Gr*inv; gradA[iKS] = -pq*Gr*inv*inv; A_all[iKS] = A; }
        __syncthreads();
    }
}

// ---- pass2 chunk reverse totals: cGB[n,s,g,d]=sum gB, cGA[n,s,g]=sum gradA ----
__global__ void racebwd_p2_totals(const float *__restrict__ pQ, const float *__restrict__ GO,
                                  const float *__restrict__ A_all, const float *__restrict__ gradA,
                                  float *__restrict__ cGB, float *__restrict__ cGA,
                                  int N, int T, int S, int D, int C, int G, float eps)
{
    int g = blockIdx.x, s = blockIdx.y, n = blockIdx.z, d = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T); if (t0 >= T) return;
    float lgb = 0.f, lga = 0.f; bool hd = d < D;
    for (int t = t0; t < t1; ++t)
    {
        size_t iKS = ((size_t)n*T+t)*S+s; float pq = pQ[iKS]; float inv = 1.f/(A_all[iKS]+eps);
        lga += gradA[iKS];
        if (hd) lgb += pq*GO[((size_t)n*T+t)*(size_t)D+d]*inv;
    }
    if (hd) cGB[(((size_t)n*S+s)*G+g)*(size_t)D+d] = lgb;
    if (d == 0) cGA[((size_t)n*S+s)*G+g] = lga;
}

// ---- reverse exclusive scan over g (in place): off[g] = sum_{g'>g} c[g'] ----
__global__ void racebwd_scan_rev(float *__restrict__ cGB, float *__restrict__ cGA, int N, int S, int D, int G)
{
    int s = blockIdx.x, n = blockIdx.y, tid = threadIdx.x;
    if (tid == 0) { float r = 0.f; for (int g = G-1; g >= 0; --g) { size_t i = ((size_t)n*S+s)*G+g; float v = cGA[i]; cGA[i] = r; r += v; } }
    for (int d = tid; d < D; d += blockDim.x) { float r = 0.f; for (int g = G-1; g >= 0; --g) { size_t i = (((size_t)n*S+s)*G+g)*(size_t)D+d; float v = cGB[i]; cGB[i] = r; r += v; } }
}

// ---- pass2 gradProbsK readout (reverse): gradProbsK = gAn + sum_d gBn[d]*v ----
__global__ void racebwd_p2_kq(const float *__restrict__ pQ, const float *__restrict__ V2, const float *__restrict__ GO,
                              const float *__restrict__ A_all, const float *__restrict__ gradA,
                              const float *__restrict__ sgAoff, const float *__restrict__ sgBoff,
                              float *__restrict__ gradProbsK,
                              int N, int T, int S, int D, int C, int G, float eps)
{
    int g = blockIdx.x, s = blockIdx.y, n = blockIdx.z, tid = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T); if (t0 >= T) return;
    extern __shared__ float shGB[];              // [D]
    __shared__ float shGA;
    for (int d = tid; d < D; d += blockDim.x) shGB[d] = sgBoff[(((size_t)n*S+s)*G+g)*(size_t)D+d];
    if (tid == 0) shGA = sgAoff[((size_t)n*S+s)*G+g];
    __syncthreads();
    for (int t = t1 - 1; t >= t0; --t)
    {
        size_t iKS = ((size_t)n*T+t)*S+s; float pq = pQ[iKS]; float inv = 1.f/(A_all[iKS]+eps);
        if (tid == 0) shGA += gradA[iKS];
        size_t bVD = ((size_t)n*T+t)*(size_t)D; float sv = 0.f;
        for (int d = tid; d < D; d += blockDim.x) { float gB = pq*GO[bVD+d]*inv; float gg = shGB[d]+gB; shGB[d] = gg; sv += gg*V2[bVD+d]; }
        float sgbv = block_reduce_sum(sv);
        if (tid == 0) gradProbsK[iKS] = shGA + sgbv;
        __syncthreads();
    }
}

// ---- pass2 gradV readout (reverse): gradV[t,d] = sum_s pk*gBn[s]  (block=S threads) ----
__global__ void racebwd_p2_v(const float *__restrict__ pK, const float *__restrict__ pQ, const float *__restrict__ GO,
                             const float *__restrict__ A_all, const float *__restrict__ sgBoff,
                             float *__restrict__ gradV,
                             int N, int T, int S, int D, int C, int G, float eps)
{
    int g = blockIdx.x, n = blockIdx.y, d = blockIdx.z, s = threadIdx.x;
    int t0 = g * C, t1 = min(t0 + C, T); if (t0 >= T || s >= S) return;
    extern __shared__ float red[];               // [S]
    float gBn = sgBoff[(((size_t)n*S+s)*G+g)*(size_t)D+d];
    for (int t = t1 - 1; t >= t0; --t)
    {
        size_t iKS = ((size_t)n*T+t)*S+s; float inv = 1.f/(A_all[iKS]+eps);
        gBn += pQ[iKS]*GO[((size_t)n*T+t)*(size_t)D+d]*inv;
        red[s] = pK[iKS]*gBn;
        __syncthreads();
        if (s == 0) { float tot = 0.f; for (int i = 0; i < S; ++i) tot += red[i]; gradV[((size_t)n*T+t)*(size_t)D+d] = tot; }
        __syncthreads();
    }
}

// ---- host wrapper: returns {gradProbsK, gradProbsQ, gradV} ----
std::vector<Tensor> race_backward(Tensor probsK, Tensor probsQ, Tensor V2, Tensor grad_out, float eps, int64_t chunk)
{
    TORCH_CHECK(probsK.is_cuda() && probsQ.is_cuda() && V2.is_cuda() && grad_out.is_cuda(), "CUDA only");
    TORCH_CHECK(probsK.scalar_type() == at::kFloat && probsQ.scalar_type() == at::kFloat && V2.scalar_type() == at::kFloat && grad_out.scalar_type() == at::kFloat, "fp32 only");
    int N = probsK.size(0), T = probsK.size(1), S = probsK.size(2), D = V2.size(2);
    TORCH_CHECK(S <= 1024, "S must be <= 1024");
    int C = (int)chunk; if (C <= 0) C = 8192; if (C > T) C = T; int G = (T + C - 1) / C;
    auto opt = torch::TensorOptions().device(probsK.device()).dtype(at::kFloat);

    auto cB = torch::empty({N, S, G, D}, opt), cA = torch::empty({N, S, G}, opt);
    auto A_all = torch::empty({N, T, S}, opt), gradA = torch::empty({N, T, S}, opt);
    auto gradProbsQ = torch::empty({N, T, S}, opt), gradProbsK = torch::empty({N, T, S}, opt);
    auto gradV = torch::empty_like(grad_out);
    auto cGB = torch::empty({N, S, G, D}, opt), cGA = torch::empty({N, S, G}, opt);

    int blk = pad_to_warp(D);
    dim3 gGSN((unsigned)G, (unsigned)S, (unsigned)N), gSN((unsigned)S, (unsigned)N, 1);
    size_t shD = (size_t)D * sizeof(float);
    const float *pK = probsK.data_ptr<float>(), *pQ = probsQ.data_ptr<float>(),
                *v = V2.data_ptr<float>(), *go = grad_out.data_ptr<float>();

    // pass1
    racebwd_p1_totals<<<gGSN, blk>>>(pK, v, cB.data_ptr<float>(), cA.data_ptr<float>(), N, T, S, D, C, G);
    racebwd_scan_fwd<<<gSN, 256>>>(cB.data_ptr<float>(), cA.data_ptr<float>(), N, S, D, G);
    racebwd_p1_readout<<<gGSN, blk, shD>>>(pK, pQ, v, go, cA.data_ptr<float>(), cB.data_ptr<float>(),
        gradProbsQ.data_ptr<float>(), gradA.data_ptr<float>(), A_all.data_ptr<float>(), N, T, S, D, C, G, eps);

    // pass2 shared totals + reverse scan
    racebwd_p2_totals<<<gGSN, blk>>>(pQ, go, A_all.data_ptr<float>(), gradA.data_ptr<float>(),
        cGB.data_ptr<float>(), cGA.data_ptr<float>(), N, T, S, D, C, G, eps);
    racebwd_scan_rev<<<gSN, 256>>>(cGB.data_ptr<float>(), cGA.data_ptr<float>(), N, S, D, G);

    // pass2 readouts (share cGB/cGA suffix offsets)
    racebwd_p2_kq<<<gGSN, blk, shD>>>(pQ, v, go, A_all.data_ptr<float>(), gradA.data_ptr<float>(),
        cGA.data_ptr<float>(), cGB.data_ptr<float>(), gradProbsK.data_ptr<float>(), N, T, S, D, C, G, eps);
    dim3 gGND((unsigned)G, (unsigned)N, (unsigned)D); dim3 bS((unsigned)S, 1, 1);
    size_t shS = (size_t)S * sizeof(float);
    racebwd_p2_v<<<gGND, bS, shS>>>(pK, pQ, go, A_all.data_ptr<float>(), cGB.data_ptr<float>(),
        gradV.data_ptr<float>(), N, T, S, D, C, G, eps);

    return {gradProbsK, gradProbsQ, gradV};
}
