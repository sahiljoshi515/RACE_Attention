# Codebase overview (auto-generated inventory)

Repo: github.com/sahiljoshi515/RACE_Attention  |  branch: `feat/vllm-race-attention-backend`  |  HEAD: `c620cdc`
Regenerate: scripts/build_codebase.sh  (the narrative codebase/* pages are hand-curated)

## Directory map (excludes arXiv source + docs/)

```
.
kernels
  cpu
  gpu
misc
notebooks
scaling
```

## Source files (code only)

```
kernels/cpu/linear_pref.cpp
kernels/cpu/race_ext.py
kernels/cpu/race_pref.cpp
kernels/cpu/setup.py
kernels/gpu/backward_kernels.cu
kernels/gpu/forward_kernel.cu
kernels/gpu/race_cuda_binding.cpp
kernels/gpu/race_cuda_build.py
misc/arxiv_64K.py
misc/classification.py
misc/food-101.py
misc/gpt.py
misc/lm.py
misc/mlm.py
misc/race.py
misc/vit.py
scaling/benchmark_time.py
scaling/race_causal_cuda.py
scaling/race_common.py
scaling/race_torch_cumsum.py
scaling/test_kernels.py
```

## Public symbol index

Python classes / functions (kernels/, misc/, scaling/):
```
scaling/race_common.py : 14:def build_planes_protos(d_k, Kbits, L, M, device="cuda", share_planes=True,
scaling/race_common.py : 36:def soft_hash_probs(Q, K, V, planes_T, protos_T, L, Kbits, M, share_planes=True):
scaling/race_common.py : 49:    def packM(Z):
scaling/race_common.py : 76:def race_prefix_ref(probsK, probsQ, V2, eps=1e-6):
scaling/benchmark_time.py : 14:def softmax_attention(Q, K, V, eps=1e-6):
scaling/benchmark_time.py : 24:def flash_attention(Q, K, V, eps=1e-6):
scaling/benchmark_time.py : 52:def angular_attention(Q, K, V, eps=1e-6, exponent=8.0):
scaling/benchmark_time.py : 62:def linformer_attention(Q, K, V, k_proj_dim=128, eps=1e-6, Ek=None, Ev=None):
scaling/benchmark_time.py : 77:class BatchedACE(nn.Module):
scaling/benchmark_time.py : 83:    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
scaling/benchmark_time.py : 106:    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
scaling/benchmark_time.py : 168:def race_attention(Q, K, V, ace_module: BatchedACE, eps=1e-6):
scaling/benchmark_time.py : 173:    def pack(Z): return Z.permute(0,2,1,3).unsqueeze(0).expand(M, -1, -1, -1, -1)  # [M,B,T,H,D]
scaling/benchmark_time.py : 178:def linear_attention(Q, K, V, eps=1e-6):
scaling/benchmark_time.py : 192:def favorplus_features(x, proj, eps=1e-6):
scaling/benchmark_time.py : 210:def linear_attention_favorplus(Q, K, V, proj, eps=1e-6):
scaling/benchmark_time.py : 235:def orthogonal_random_matrix(M, D, device=None, dtype=None):
scaling/benchmark_time.py : 249:def make_favorplus_projections(H, M, D, device=None, dtype=None, orthogonal=True):
scaling/benchmark_time.py : 263:def alloc_qkv(B,H,T,D,dtype,device):
scaling/benchmark_time.py : 269:def median_ms(vals):
scaling/benchmark_time.py : 273:def is_cuda_device(dev: str) -> bool:
scaling/benchmark_time.py : 277:def maybe_autocast(device, dtype):
scaling/benchmark_time.py : 284:def bench_one(method, T, B, H, D, device, dtype, warmup, iters, knobs):
scaling/benchmark_time.py : 305:    def step():
scaling/benchmark_time.py : 353:def main():
scaling/test_kernels.py : 30:def relerr(a, b):
scaling/test_kernels.py : 34:def maxabs(a, b):
scaling/test_kernels.py : 38:def check(name, got, ref, tol, abs_floor=2e-5):
scaling/test_kernels.py : 54:def make_inputs(N, T, L, R, D, seed, dist="softmax"):
scaling/test_kernels.py : 69:def run_case(N, T, L, Kbits, D, seed=0, eps=1e-6, dist="softmax", fwd_tol=2e-5, grad_tol=3e-3):
scaling/test_kernels.py : 107:def run_module_e2e(H, T, L, Kbits, D, B=1, seed=0, tol=2e-3):
scaling/test_kernels.py : 132:def main():
scaling/race_causal_cuda.py : 26:def _ext():
scaling/race_causal_cuda.py : 33:def fwd_chunk(T):
scaling/race_causal_cuda.py : 47:class RaceCausalFn(Function):
scaling/race_causal_cuda.py : 51:    def forward(ctx, probsK, probsQ, V2, eps):
scaling/race_causal_cuda.py : 64:    def backward(ctx, grad_out):
scaling/race_causal_cuda.py : 72:def race_cuda_fused(probsK, probsQ, V2, eps=1e-6):
scaling/race_causal_cuda.py : 76:class RaceCausalCuda(nn.Module):
scaling/race_causal_cuda.py : 82:    def __init__(self, d_k, Kbits, L, M, device="cuda", share_planes=True,
scaling/race_causal_cuda.py : 94:    def forward(self, Q, K, V):
scaling/race_torch_cumsum.py : 14:class RaceCumsumCausal(nn.Module):
scaling/race_torch_cumsum.py : 15:    def __init__(self, d_k, Kbits, L, M, device="cuda", share_planes=True,
scaling/race_torch_cumsum.py : 27:    def forward(self, Q, K, V):
kernels/cpu/setup.py : 43:def rfa_prefix_mean_ref(probsK, V, eps=1e-6):
kernels/cpu/setup.py : 50:def median_time(fn, *args, warmup=3, repeat=10):
kernels/cpu/setup.py : 62:def bench_once(N=32, T=512, L=4, R=4, D=64, eps=1e-6, device="cpu"):
kernels/cpu/setup.py : 117:class RFAPrefixMeanFlatFn(torch.autograd.Function):
kernels/cpu/setup.py : 119:    def forward(ctx, probsK_flat, V_flat, eps: float):
kernels/cpu/setup.py : 128:    def backward(ctx, gradE):
kernels/cpu/setup.py : 134:def rfa_prefix_mean_flat_ref(w, v, eps=1e-6):
kernels/cpu/setup.py : 169:def relerr(a, b, eps=1e-12):
kernels/gpu/race_cuda_build.py : 17:def load_ext(verbose=True):
misc/gpt.py : 8:class MultiHeadAttention(nn.Module):
misc/gpt.py : 9:    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
misc/gpt.py : 29:    def forward(self, x):
misc/gpt.py : 68:class TransformerBlock(nn.Module):
misc/gpt.py : 69:    def __init__(self, cfg):
misc/gpt.py : 87:    def forward(self, x):
misc/gpt.py : 104:class GPTModel(nn.Module):
misc/gpt.py : 105:    def __init__(self, cfg):
misc/gpt.py : 119:    def forward(self, in_idx):
misc/gpt.py : 133:class AngularAttention(nn.Module):
misc/gpt.py : 134:    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
misc/gpt.py : 155:    def forward(self, x, use_sketches=False):
misc/gpt.py : 199:class AngularBlock(nn.Module):
misc/gpt.py : 200:    def __init__(self, cfg):
misc/gpt.py : 218:    def forward(self, x):
misc/gpt.py : 235:class AngularModel(nn.Module):
misc/gpt.py : 236:    def __init__(self, cfg):
misc/gpt.py : 250:    def forward(self, in_idx):
misc/lm.py : 37:class LMModel(nn.Module):
misc/lm.py : 38:    def __init__(self, cfg, attn_type="gpt", device="cpu"):
misc/lm.py : 67:    def forward(self, x):
misc/lm.py : 80:class GPTDatasetV1(Dataset):
misc/lm.py : 81:    def __init__(self, txt, tokenizer, max_length, stride):
misc/lm.py : 95:    def __len__(self):
misc/lm.py : 98:    def __getitem__(self, idx):
misc/lm.py : 102:def create_dataloader_v1(txt, batch_size, max_length, 
misc/lm.py : 126:class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
misc/lm.py : 131:    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
misc/lm.py : 136:    def get_lr(self):
misc/lm.py : 149:def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, cfg, grad_accum_steps=1):
misc/lm.py : 166:    def _log(fp, msg):
misc/lm.py : 260:def load_wikitext():
misc/lm.py : 298:def load_ptb(context_len, batch_size=16, stride=None):
misc/lm.py : 306:    def col_name(split):
misc/lm.py : 311:    def join_lines(split):
misc/lm.py : 353:def start_experiment():
misc/vit.py : 33:def get_data(cfg):
misc/vit.py : 67:class PatchEmbedding(nn.Module):
misc/vit.py : 68:    def __init__(self, cfg):
misc/vit.py : 72:    def forward(self, x):
misc/vit.py : 79:class MultiHeadAttention(nn.Module):
misc/vit.py : 80:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
misc/vit.py : 93:    def forward(self, x):
misc/vit.py : 103:class TransformerArchitecture(nn.Module):
misc/vit.py : 104:    def __init__(self, cfg):
misc/vit.py : 115:    def forward(self, x):
misc/vit.py : 124:class BatchedACE(nn.Module):
misc/vit.py : 130:    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
misc/vit.py : 154:    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
misc/vit.py : 218:    def sync_from_soft(self, ace_soft):  # ace_soft is BatchedACE
misc/vit.py : 230:class RACEAttention(nn.Module):
misc/vit.py : 231:    def __init__(self, d_in, d_out, dropout,
misc/vit.py : 248:    def forward(self, x):
misc/vit.py : 258:        def pack(Z):
misc/vit.py : 276:class RACEBlock(nn.Module):
misc/vit.py : 277:    def __init__(self, cfg, device='cpu'):
misc/vit.py : 294:    def forward(self, x):
misc/vit.py : 306:def favorplus_features(x, proj, eps=1e-6):
misc/vit.py : 326:class FavorPlusAttention(nn.Module):
misc/vit.py : 332:    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
misc/vit.py : 355:    def forward(self, x):
misc/vit.py : 401:class PerformerBlock(nn.Module):
misc/vit.py : 405:    def __init__(self, cfg):
misc/vit.py : 424:    def forward(self, x):
misc/vit.py : 439:class LinearAttention(nn.Module):
misc/vit.py : 440:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
misc/vit.py : 453:    def kernel(self, x):
misc/vit.py : 457:    def forward(self, x):
misc/vit.py : 483:class LinearBlock(nn.Module):
misc/vit.py : 484:    def __init__(self, cfg):
misc/vit.py : 500:    def forward(self, x):
misc/vit.py : 510:class AngularAttention(nn.Module):
misc/vit.py : 511:    def __init__(self, d, h, drop, qkv_bias=False):
misc/vit.py : 520:    def forward(self, x):
misc/vit.py : 533:class AngularBlock(nn.Module):
misc/vit.py : 534:    def __init__(self, cfg):
misc/vit.py : 547:    def forward(self, x):
misc/vit.py : 559:class LinformerAttention(nn.Module):
misc/vit.py : 568:    def __init__(
misc/vit.py : 599:    def forward(self, x: torch.Tensor) -> torch.Tensor:
misc/vit.py : 632:class LinformerBlock(nn.Module):
misc/vit.py : 637:    def __init__(self, cfg):
misc/vit.py : 656:    def forward(self, x):
misc/vit.py : 670:class VisionTransformer(nn.Module):
misc/vit.py : 671:    def __init__(self, cfg, attn_type, device='cpu'):
misc/vit.py : 706:    def forward(self, x):
misc/vit.py : 717:class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
misc/vit.py : 722:    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
misc/vit.py : 727:    def get_lr(self):
misc/vit.py : 740:def train_model_simple(
misc/vit.py : 773:    def _log(fp, msg):
misc/vit.py : 887:def start_experiment():
misc/arxiv_64K.py : 85:def basic_english_tokenizer(text: str) -> List[str]:
misc/arxiv_64K.py : 102:def eda_random_deletion(tokens, p=0.05):
misc/arxiv_64K.py : 108:def eda_random_swap(tokens, n_swaps=3):
misc/arxiv_64K.py : 120:def print_length_stats(arr: np.ndarray, name: str, thresholds=()):
misc/arxiv_64K.py : 141:def make_balanced_long_examples(split, desired_total, min_len, name="train", seed=SEED):
misc/arxiv_64K.py : 216:def pack_examples_streaming(
misc/arxiv_64K.py : 279:class ArxivDataset(Dataset):
misc/arxiv_64K.py : 280:    def __init__(self, examples, max_len, stoi, pad_idx=0, unk_idx=1, augment=False):
misc/arxiv_64K.py : 288:    def __len__(self):
misc/arxiv_64K.py : 291:    def __getitem__(self, idx):
misc/arxiv_64K.py : 308:    def collate_fn(self, batch):
misc/arxiv_64K.py : 317:def compute_effective_lengths_from_loader(dl, num_batches=100):
misc/arxiv_64K.py : 328:def print_effective_length_stats(arr: np.ndarray, max_len: int, thresholds=()):
misc/arxiv_64K.py : 404:def packed_lengths(examples):
misc/arxiv_64K.py : 510:class MultiHeadAttention(nn.Module):
misc/arxiv_64K.py : 512:    def __init__(self, d, h, drop, qkv_bias=False):
misc/arxiv_64K.py : 522:    def forward(self, x, mask):
misc/arxiv_64K.py : 548:class AngularAttention(nn.Module):
misc/arxiv_64K.py : 550:    def __init__(self, d, h, drop, qkv_bias=False):
misc/arxiv_64K.py : 560:    def forward(self, x, mask):
misc/arxiv_64K.py : 584:class BatchedACE(nn.Module):
misc/arxiv_64K.py : 586:    def __init__(self, d_k, K, L, M, device="cpu", share_planes=False):
misc/arxiv_64K.py : 605:    def forward(self, Khf, Vhf, Qhf, eps=1e-6):
misc/arxiv_64K.py : 652:class RACEAttention(nn.Module):
misc/arxiv_64K.py : 653:    def __init__(self, d, h, drop, K, L, M, qkv_bias=False, device="cpu"):
misc/arxiv_64K.py : 664:    def forward(self, x, mask):
misc/arxiv_64K.py : 676:        def pack(z):
misc/arxiv_64K.py : 685:def favorplus_features(x, proj, eps=1e-6):
misc/arxiv_64K.py : 693:class FavorPlusAttention(nn.Module):
misc/arxiv_64K.py : 694:    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
misc/arxiv_64K.py : 713:    def forward(self, x, mask=None):
misc/arxiv_64K.py : 748:class LinearAttention(nn.Module):
misc/arxiv_64K.py : 749:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
misc/arxiv_64K.py : 762:    def kernel(self, x):
misc/arxiv_64K.py : 765:    def forward(self, x, mask=None):
misc/arxiv_64K.py : 793:class LinformerAttention(nn.Module):
misc/arxiv_64K.py : 794:    def __init__(self, d, dropout, num_heads, qkv_bias, k_proj_dim, max_seq_len):
misc/arxiv_64K.py : 814:    def forward(self, x, mask=None):
misc/arxiv_64K.py : 846:class SoftmaxBlock(nn.Module):
misc/arxiv_64K.py : 847:    def __init__(self, cfg):
misc/arxiv_64K.py : 863:    def forward(self, x, mask):
misc/arxiv_64K.py : 875:class AngularBlock(nn.Module):
misc/arxiv_64K.py : 876:    def __init__(self, cfg):
misc/arxiv_64K.py : 892:    def forward(self, x, mask):
misc/arxiv_64K.py : 904:class RACEBlock(nn.Module):
misc/arxiv_64K.py : 905:    def __init__(self, cfg, device=DEVICE):
misc/arxiv_64K.py : 925:    def forward(self, x, mask):
misc/arxiv_64K.py : 937:class LinearBlock(nn.Module):
misc/arxiv_64K.py : 938:    def __init__(self, cfg):
misc/arxiv_64K.py : 956:    def forward(self, x, mask):
misc/arxiv_64K.py : 968:class LinformerBlock(nn.Module):
misc/arxiv_64K.py : 969:    def __init__(self, cfg):
misc/arxiv_64K.py : 993:    def forward(self, x, mask):
misc/arxiv_64K.py : 1005:class PerformerBlock(nn.Module):
misc/arxiv_64K.py : 1006:    def __init__(self, cfg):
misc/arxiv_64K.py : 1028:    def forward(self, x, mask):
misc/arxiv_64K.py : 1043:class TextTransformerClassifier(nn.Module):
misc/arxiv_64K.py : 1044:    def __init__(self, cfg, attn_type: str):
misc/arxiv_64K.py : 1076:    def forward(self, x, mask):
misc/arxiv_64K.py : 1091:class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
misc/arxiv_64K.py : 1092:    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
misc/arxiv_64K.py : 1097:    def get_lr(self):
misc/arxiv_64K.py : 1109:def train_model_simple(
misc/arxiv_64K.py : 1133:    def _log(fp, msg):
misc/arxiv_64K.py : 1264:def run_experiment(attn_types, cfg):
misc/food-101.py : 47:def _get_labels(ds):
misc/food-101.py : 54:def _balanced_subset_fixed_total(ds, class_ids, total, seed=0):
misc/food-101.py : 86:def get_data_food101(
misc/food-101.py : 158:class PatchEmbedding(nn.Module):
misc/food-101.py : 159:    def __init__(self, cfg):
misc/food-101.py : 163:    def forward(self, x):
misc/food-101.py : 170:class MultiHeadAttention(nn.Module):
misc/food-101.py : 171:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
misc/food-101.py : 184:    def forward(self, x):
misc/food-101.py : 203:class TransformerArchitecture(nn.Module):
misc/food-101.py : 204:    def __init__(self, cfg):
misc/food-101.py : 215:    def forward(self, x):
misc/food-101.py : 224:class BatchedACE(nn.Module):
misc/food-101.py : 230:    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
misc/food-101.py : 254:    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
misc/food-101.py : 318:    def sync_from_soft(self, ace_soft):  # ace_soft is BatchedACE
misc/food-101.py : 330:class RACEAttention(nn.Module):
misc/food-101.py : 331:    def __init__(self, d_in, d_out, dropout,
misc/food-101.py : 348:    def forward(self, x):
misc/food-101.py : 358:        def pack(Z):
misc/food-101.py : 376:class RACEBlock(nn.Module):
misc/food-101.py : 377:    def __init__(self, cfg, device='cpu'):
misc/food-101.py : 394:    def forward(self, x):
misc/food-101.py : 406:def favorplus_features(x, proj, eps=1e-6):
misc/food-101.py : 426:class FavorPlusAttention(nn.Module):
misc/food-101.py : 432:    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
misc/food-101.py : 455:    def forward(self, x):
misc/food-101.py : 501:class PerformerBlock(nn.Module):
misc/food-101.py : 505:    def __init__(self, cfg):
misc/food-101.py : 524:    def forward(self, x):
misc/food-101.py : 539:class LinearAttention(nn.Module):
misc/food-101.py : 540:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
misc/food-101.py : 553:    def kernel(self, x):
misc/food-101.py : 557:    def forward(self, x):
misc/food-101.py : 583:class LinearBlock(nn.Module):
misc/food-101.py : 584:    def __init__(self, cfg):
misc/food-101.py : 600:    def forward(self, x):
misc/food-101.py : 610:class AngularAttention(nn.Module):
misc/food-101.py : 611:    def __init__(self, d, h, drop, qkv_bias=False):
misc/food-101.py : 620:    def forward(self, x):
misc/food-101.py : 633:class AngularBlock(nn.Module):
misc/food-101.py : 634:    def __init__(self, cfg):
misc/food-101.py : 647:    def forward(self, x):
misc/food-101.py : 659:class LinformerAttention(nn.Module):
misc/food-101.py : 668:    def __init__(
misc/food-101.py : 699:    def forward(self, x: torch.Tensor) -> torch.Tensor:
misc/food-101.py : 732:class LinformerBlock(nn.Module):
misc/food-101.py : 737:    def __init__(self, cfg):
misc/food-101.py : 756:    def forward(self, x):
misc/food-101.py : 770:class VisionTransformer(nn.Module):
misc/food-101.py : 771:    def __init__(self, cfg, attn_type, device='cuda'):
misc/food-101.py : 813:    def forward(self, x):
misc/food-101.py : 823:class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
misc/food-101.py : 828:    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
misc/food-101.py : 833:    def get_lr(self):
misc/food-101.py : 846:def train_model_simple(
misc/food-101.py : 879:    def _log(fp, msg):
misc/food-101.py : 994:def start_experiment():
misc/mlm.py : 37:class MultiHeadAttention(nn.Module):
misc/mlm.py : 38:    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
misc/mlm.py : 50:    def forward(self, x):
misc/mlm.py : 60:class TransformerBlock(nn.Module):
misc/mlm.py : 61:    def __init__(self, cfg):
misc/mlm.py : 78:    def forward(self, x):
misc/mlm.py : 90:class AngularAttention(nn.Module):
misc/mlm.py : 91:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
misc/mlm.py : 103:    def forward(self, x):
misc/mlm.py : 121:class AngularBlock(nn.Module):
misc/mlm.py : 122:    def __init__(self, cfg):
misc/mlm.py : 138:    def forward(self, x):
misc/mlm.py : 149:class BatchedACE(nn.Module):
misc/mlm.py : 155:    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
misc/mlm.py : 178:    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
misc/mlm.py : 240:class LinearAttention(nn.Module):
misc/mlm.py : 241:    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
misc/mlm.py : 254:    def kernel(self, x):
misc/mlm.py : 258:    def forward(self, x):
misc/mlm.py : 284:class LinearBlock(nn.Module):
misc/mlm.py : 285:    def __init__(self, cfg):
misc/mlm.py : 301:    def forward(self, x):
misc/mlm.py : 311:class RACEAttention(nn.Module):
misc/mlm.py : 312:    def __init__(self, d_in, d_out, dropout,
misc/mlm.py : 327:    def forward(self, x):
misc/mlm.py : 337:        def pack(Z):
misc/mlm.py : 357:class RACEBlock(nn.Module):
misc/mlm.py : 358:    def __init__(self, cfg, device='cpu'):
misc/mlm.py : 375:    def forward(self, x):
misc/mlm.py : 387:class LinformerAttention(nn.Module):
misc/mlm.py : 396:    def __init__(
misc/mlm.py : 427:    def forward(self, x: torch.Tensor) -> torch.Tensor:
misc/mlm.py : 460:class LinformerBlock(nn.Module):
misc/mlm.py : 465:    def __init__(self, cfg):
misc/mlm.py : 484:    def forward(self, x):
misc/mlm.py : 498:def favorplus_features(x, proj, eps=1e-6):
misc/mlm.py : 518:class FavorPlusAttention(nn.Module):
misc/mlm.py : 524:    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
misc/mlm.py : 547:    def forward(self, x):
misc/mlm.py : 593:class PerformerBlock(nn.Module):
misc/mlm.py : 597:    def __init__(self, cfg):
misc/mlm.py : 616:    def forward(self, x):
misc/mlm.py : 635:class LMModel(nn.Module):
misc/mlm.py : 636:    def __init__(self, cfg, attn_type="gpt", device="cpu"):
misc/mlm.py : 663:    def forward_hidden(self, x):
misc/mlm.py : 673:    def forward(self, x):
misc/mlm.py : 681:def load_small_tinystories():
misc/mlm.py : 686:    def tokenize_function(examples):
misc/mlm.py : 691:    def group_texts(examples):
misc/mlm.py : 706:    def tuple_collate_fn(features):
misc/mlm.py : 723:class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
misc/mlm.py : 728:    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
misc/mlm.py : 733:    def get_lr(self):
misc/mlm.py : 746:def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, cfg, grad_accum_steps=1):
misc/mlm.py : 772:    def _log(fp, msg):
misc/mlm.py : 900:def start_experiment():
misc/classification.py : 41:def basic_english_tokenizer(text: str) -> list[str]:
misc/classification.py : 56:def eda_random_deletion(tokens, p=0.05):
misc/classification.py : 62:def eda_random_swap(tokens, n_swaps=3):
misc/classification.py : 90:class AugmentedIMDB(Dataset):
misc/classification.py : 91:    def __init__(self, examples, max_len, augment=False):
misc/classification.py : 96:    def __len__(self):
misc/classification.py : 99:    def __getitem__(self, idx):
misc/classification.py : 112:    def collate_fn(self, batch):
misc/classification.py : 122:def make_long_subsets(examples):
misc/classification.py : 179:def get_data():
misc/classification.py : 204:class MultiHeadAttention(nn.Module):
misc/classification.py : 205:    def __init__(self, d, h, drop, qkv_bias=False):
misc/classification.py : 214:    def forward(self, x, mask):
misc/classification.py : 230:class AngularAttention(nn.Module):
misc/classification.py : 231:    def __init__(self, d, h, drop, qkv_bias=False):
misc/classification.py : 240:    def forward(self, x, mask):
misc/classification.py : 257:class BatchedACE(nn.Module):
misc/classification.py : 258:    def __init__(self, d_k, K, L, M):
misc/classification.py : 269:    def forward(self, Kh, Vh, Qh):
misc/classification.py : 318:class RACEAttention(nn.Module):
misc/classification.py : 320:    def __init__(self, d, h, drop, M=2, K=3, L=2, qkv_bias=False):
misc/classification.py : 331:    def forward(self, x, mask):
misc/classification.py : 343:        def pack(z):
misc/classification.py : 352:class RACEBlock(nn.Module):
misc/classification.py : 353:    def __init__(self, cfg, device='cuda'):
misc/classification.py : 373:    def forward(self, x, pad_mask):
misc/classification.py : 385:class TransformerBlock(nn.Module):
misc/classification.py : 387:    def __init__(self, cfg):
misc/classification.py : 399:    def forward(self, x, pad_mask):
misc/classification.py : 412:class AngularBlock(nn.Module):
misc/classification.py : 414:    def __init__(self, cfg):
misc/classification.py : 427:    def forward(self, x, pad_mask):
misc/classification.py : 445:class Classifier(nn.Module):
misc/classification.py : 446:    def __init__(self, cfg, kind):
misc/classification.py : 467:    def forward(self, x, mask):
misc/classification.py : 479:def run_experiment(attn_types, epochs=5, lr=1e-5, wd=5e-05):
misc/race.py : 9:class RACEPrefixMeanFlatFn(Function):
misc/race.py : 11:    def forward(ctx, probsK_flat: torch.Tensor, V_flat: torch.Tensor, eps: float):
misc/race.py : 18:    def backward(ctx, gradE_flat):
misc/race.py : 26:class BatchedACE(nn.Module):
misc/race.py : 33:    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = True):
misc/race.py : 63:    def forward(self, Khf, Vhf, Qhf):
misc/race.py : 144:class RACEAttention(nn.Module):
misc/race.py : 146:    def __init__(self, d, h, K, L, M, drop=0.1,
misc/race.py : 160:    def forward(self, x):
misc/race.py : 176:class RACEBlock(nn.Module):
misc/race.py : 177:    def __init__(self, cfg, device='cpu'):
misc/race.py : 198:    def forward(self, x):
```

CUDA kernels (__global__) and pybind exports (m.def):
```
kernels/gpu/backward_kernels.cu:51:__global__ void racebwd_p1_totals(const float *__restrict__ pK, const float *__restrict__ V2,
kernels/gpu/backward_kernels.cu:64:__global__ void racebwd_scan_fwd(float *__restrict__ cB, float *__restrict__ cA, int N, int S, int D, int G)
kernels/gpu/backward_kernels.cu:72:__global__ void racebwd_p1_readout(const float *__restrict__ pK, const float *__restrict__ pQ,
kernels/gpu/backward_kernels.cu:100:__global__ void racebwd_p2_totals(const float *__restrict__ pQ, const float *__restrict__ GO,
kernels/gpu/backward_kernels.cu:119:__global__ void racebwd_scan_rev(float *__restrict__ cGB, float *__restrict__ cGA, int N, int S, int D, int G)
kernels/gpu/backward_kernels.cu:127:__global__ void racebwd_p2_kq(const float *__restrict__ pQ, const float *__restrict__ V2, const float *__restrict__ GO,
kernels/gpu/backward_kernels.cu:153:__global__ void racebwd_p2_v(const float *__restrict__ pK, const float *__restrict__ pQ, const float *__restrict__ GO,
kernels/gpu/forward_kernel.cu:26:__global__ void racefwd_phase1(
kernels/gpu/forward_kernel.cu:50:__global__ void racefwd_phase2(
kernels/gpu/forward_kernel.cu:75:__global__ void racefwd_phase3(
kernels/gpu/race_cuda_binding.cpp:18:    m.def("race_fused_fwd", &race_fused_fwd,
kernels/gpu/race_cuda_binding.cpp:21:    m.def("race_backward", &race_backward,
kernels/cpu/linear_pref.cpp:217:    m.def(
kernels/cpu/linear_pref.cpp:222:    m.def(
kernels/cpu/race_pref.cpp:224:    m.def("race_prefix_mean_flat", &race_prefix_mean_flat, "Race prefix mean (probsK[NS,T], V[NS,T,D])");
kernels/cpu/race_pref.cpp:225:    m.def("race_prefix_mean_flat_bw", &race_prefix_mean_flat_bw, "Race prefix mean backward (flat)");
```

## Install & usage (from README.md)

See README.md. Quickstart: `pip install -r requirements.txt`; notebooks/ for runnable
examples; CPU kernels build via kernels/cpu (JIT load_ext); CUDA kernels JIT-compile via
kernels/gpu/race_cuda_build.py:load_ext() targeting sm_90 (Hopper/H200).

---
Source: live repo tree at HEAD c620cdc. See the curated pages codebase/{cpu-kernels,gpu-kernels,python-api,scaling-module,tests-benchmarks,training-scripts,vllm-backend} for narrative detail.
