// ============================================================================
// Fused single-token RACE decode step (fp32, sm_90 / H200).
//
// One decode token per row n (N = B*num_heads rows). For each row this kernel
// fuses the soft-hash of q and k, the running-prefix state update (A, B updated
// IN PLACE), and the readout:
//
//   soft-hash of x in {q,k}:
//     proj[j]      = sum_d x[d] * planes_T[d,j]            j in [0,LB), LB=L*Kbits
//     t[l,b]       = tanh(proj[l*Kbits+b]) / scale
//     logit[l,r]   = sum_b t[l,b] * protos_T[b,r]          r in [0,R), R=1<<Kbits
//     p[l*R+r]     = softmax_r(logit[l,:])                 (max-subtracted)
//   -> pk = soft-hash(k), pq = soft-hash(q),  s = l*R + r in [0,S), S = L*R
//
//   state update (each (s,d) cell touched by exactly one thread -> no atomics):
//     A[n,s]   += pk[s]                                    (added ONCE per s)
//     B[n,s,d] += pk[s] * v[n,d]
//     out[n,d]  = sum_s pq[s] * B[n,s,d] / (A[n,s] + eps)
//
// Matches distill/race_llama_attention.py (_soft_hash + the T==1 decode branch)
// to fp32 precision. Fully separate translation unit from the training kernels.
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // getCurrentCUDAStream: launch on the current
                                     // (capture) stream so the kernel is recorded into
                                     // CUDA graphs; the default-stream launch was a
                                     // no-op under torch.cuda.graph capture (-> NaN).
#include <stdint.h>

using at::Tensor;

// ---- warp/block reduce (copied from backward_kernels.cu:31-47; static here so
//      this separate translation unit keeps its own internal linkage) ----------
static __inline__ __device__ float warp_reduce_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
static __inline__ __device__ float block_reduce_sum(float v)
{
    static __shared__ float sh[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sh[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

// ---- fused decode kernel.  grid = N blocks, block = hd threads -----------------
//
// Dynamic shared memory layout (all float, contiguous in this order):
//   sh_q   [hd]
//   sh_k   [hd]
//   sh_v   [hd]
//   sh_projK [LB]
//   sh_projQ [LB]
//   sh_pk  [S]
//   sh_pq  [S]
//   sh_A   [S]
// total = (3*hd + 2*LB + 3*S) floats   (matches the host wrapper sizing).
__global__ void race_decode_kernel(
    const float *__restrict__ q,        // [N, hd]
    const float *__restrict__ k,        // [N, hd]
    const float *__restrict__ v,        // [N, hd]
    const float *__restrict__ planes_T, // [hd, LB]  row-major: planes_T[d*LB + j]
    const float *__restrict__ protos_T, // [Kbits, R] row-major: protos_T[b*R + r]
    float *__restrict__ A,              // [N, S]      (in place)
    float *__restrict__ B,              // [N, S, hd]  (in place)
    float *__restrict__ out,            // [N, hd]     (written)
    float scale, float eps,
    int N, int hd, int L, int Kbits, int R, int S, int LB)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;   // d index, in [0, blockDim.x), blockDim.x == hd

    extern __shared__ float smem[];
    float *sh_q     = smem;
    float *sh_k     = sh_q + hd;
    float *sh_v     = sh_k + hd;
    float *sh_projK = sh_v + hd;
    float *sh_projQ = sh_projK + LB;
    float *sh_pk    = sh_projQ + LB;
    float *sh_pq    = sh_pk + S;
    float *sh_A     = sh_pq + S;

    // 1. load q,k,v rows (one element per thread; blockDim.x == hd) ------------
    if (tid < hd) {
        size_t base = (size_t)n * hd + tid;
        sh_q[tid] = q[base];
        sh_k[tid] = k[base];
        sh_v[tid] = v[base];
    }
    __syncthreads();

    // 2. soft-hash projections (reduction over d). LB <= 16, loop j serially. --
    //    Every thread participates in block_reduce_sum, so all threads must run
    //    the loop the same number of times (the reduction __syncthreads()).
    for (int j = 0; j < LB; ++j) {
        float pj = (tid < hd) ? planes_T[(size_t)tid * LB + j] : 0.0f;
        float dotK = block_reduce_sum((tid < hd) ? sh_k[tid] * pj : 0.0f);
        __syncthreads();
        float dotQ = block_reduce_sum((tid < hd) ? sh_q[tid] * pj : 0.0f);
        __syncthreads();
        if (tid == 0) {
            sh_projK[j] = tanhf(dotK) / scale;
            sh_projQ[j] = tanhf(dotQ) / scale;
        }
    }
    __syncthreads();

    // 3. logits + per-L-group softmax over R contiguous entries ---------------
    //    thread s (s < S) computes its own logit for both pk and pq.
    for (int s = tid; s < S; s += blockDim.x) {
        int l = s / R;
        int r = s % R;
        const float *tK = &sh_projK[l * Kbits];
        const float *tQ = &sh_projQ[l * Kbits];
        float lk = 0.0f, lq = 0.0f;
        for (int b = 0; b < Kbits; ++b) {
            float pr = protos_T[(size_t)b * R + r];
            lk += tK[b] * pr;
            lq += tQ[b] * pr;
        }
        sh_pk[s] = lk;
        sh_pq[s] = lq;
    }
    __syncthreads();

    // per-group (l < L) max-subtract -> exp -> normalize over the R entries.
    for (int l = tid; l < L; l += blockDim.x) {
        int o = l * R;
        // softmax for pk
        float m = sh_pk[o];
        for (int r = 1; r < R; ++r) m = fmaxf(m, sh_pk[o + r]);
        float den = 0.0f;
        for (int r = 0; r < R; ++r) { float e = expf(sh_pk[o + r] - m); sh_pk[o + r] = e; den += e; }
        float inv = 1.0f / den;
        for (int r = 0; r < R; ++r) sh_pk[o + r] *= inv;
        // softmax for pq
        m = sh_pq[o];
        for (int r = 1; r < R; ++r) m = fmaxf(m, sh_pq[o + r]);
        den = 0.0f;
        for (int r = 0; r < R; ++r) { float e = expf(sh_pq[o + r] - m); sh_pq[o + r] = e; den += e; }
        inv = 1.0f / den;
        for (int r = 0; r < R; ++r) sh_pq[o + r] *= inv;
    }
    __syncthreads();

    // 4a. A update -- added EXACTLY ONCE per s (the critical race fix). --------
    for (int s = tid; s < S; s += blockDim.x) {
        float a = A[(size_t)n * S + s] + sh_pk[s];
        A[(size_t)n * S + s] = a;
        sh_A[s] = a;
    }
    __syncthreads();

    // 4b. B update + readout. Thread d owns column d for all s -> no atomics. --
    if (tid < hd) {
        float vd = sh_v[tid];
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) {
            size_t idx = ((size_t)n * S + s) * hd + tid;
            float b = B[idx] + sh_pk[s] * vd;
            B[idx] = b;
            acc += sh_pq[s] * b / (sh_A[s] + eps);
        }
        out[(size_t)n * hd + tid] = acc;
    }
}

// ---- host wrapper -------------------------------------------------------------
void race_decode_step(
    Tensor q, Tensor k, Tensor v,     // [N, hd] fp32 (post-RoPE, post-repeat_kv, single token)
    Tensor planes_T,                  // [hd, L*Kbits] fp32 (row-major)
    Tensor protos_T,                  // [Kbits, R]    fp32 (row-major)
    Tensor A,                         // [N, S]        fp32 (updated IN PLACE)
    Tensor B,                         // [N, S, hd]    fp32 (updated IN PLACE)
    Tensor out,                       // [N, hd]       fp32 (written)
    double scale, double eps,         // scale = sqrt(hd)*exp(log_temp), host-computed
    int64_t L, int64_t Kbits)         // R = 1<<Kbits, S = L*R, LB = L*Kbits
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && planes_T.is_cuda() &&
                protos_T.is_cuda() && A.is_cuda() && B.is_cuda() && out.is_cuda(),
                "CUDA only");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                v.scalar_type() == at::kFloat && planes_T.scalar_type() == at::kFloat &&
                protos_T.scalar_type() == at::kFloat && A.scalar_type() == at::kFloat &&
                B.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat,
                "fp32 only");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2 && out.dim() == 2,
                "q,k,v,out must be [N,hd]");
    TORCH_CHECK(A.dim() == 2, "A must be [N,S]");
    TORCH_CHECK(B.dim() == 3, "B must be [N,S,hd]");
    TORCH_CHECK(planes_T.dim() == 2 && protos_T.dim() == 2,
                "planes_T [hd,L*Kbits], protos_T [Kbits,R]");

    const int N  = (int)q.size(0);
    const int hd = (int)q.size(1);
    const int Kb = (int)Kbits;
    const int Lv = (int)L;
    const int R  = 1 << Kb;
    const int S  = Lv * R;
    const int LB = Lv * Kb;

    TORCH_CHECK(hd <= 1024, "hd must be <= 1024 (block size == hd)");
    TORCH_CHECK(k.size(0) == N && v.size(0) == N && out.size(0) == N &&
                A.size(0) == N && B.size(0) == N, "row count N must match across tensors");
    TORCH_CHECK(k.size(1) == hd && v.size(1) == hd && out.size(1) == hd &&
                B.size(2) == hd, "head dim hd must match across tensors");
    TORCH_CHECK(planes_T.size(0) == hd && planes_T.size(1) == LB,
                "planes_T must be [hd, L*Kbits]");
    TORCH_CHECK(protos_T.size(0) == Kb && protos_T.size(1) == R,
                "protos_T must be [Kbits, R]");
    TORCH_CHECK(A.size(1) == S && B.size(1) == S, "A/B middle dim must be S = L*R");

    // Dynamic shared memory: sh_q,sh_k,sh_v [hd] + sh_projK,sh_projQ [LB]
    //                        + sh_pk,sh_pq,sh_A [S].
    size_t shmem_bytes = (size_t)(3 * hd + 2 * LB + 3 * S) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    race_decode_kernel<<<N, hd, shmem_bytes, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        planes_T.data_ptr<float>(), protos_T.data_ptr<float>(),
        A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(),
        (float)scale, (float)eps,
        N, hd, Lv, Kb, R, S, LB);
}
