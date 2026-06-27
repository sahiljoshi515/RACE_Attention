"""Generate a clean, manager-facing PDF explaining the RACE-2.0 distillation + optimization
objective. Pure matplotlib (only PDF-capable lib available in the env). Vector, 3 pages."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

# ---- palette ----
INK   = "#10172A"   # near-black headings
GREY  = "#475569"   # body grey
TEAL  = "#0E9F8E"   # student / hybrid
BLUE  = "#2F5BD8"   # teacher / KL
ORANGE= "#E8743B"   # RACE / CE
GREEN = "#15A06B"   # hidden
LBLUE = "#E8EEFC"; LTEAL="#E1F5F2"; LOR="#FCEBE2"; LGRN="#E3F4EC"; LGREY="#F1F4F9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})

def newpage(pdf):
    fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax

def box(ax, x, y, w, h, fc, ec="none", lw=0, rad=0.018, alpha=1.0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                 fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, mutation_aspect=1.4))

def txt(ax, x, y, s, size=10.5, color=INK, weight="normal", ha="left", va="top", style="normal", z=3):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=color, weight=weight,
            ha=ha, va=va, style=style, zorder=z, wrap=True)

def arrow(ax, x0, y0, x1, y1, color=GREY, lw=2.2, z=2, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))

pdf = PdfPages("/scratch/sj157/RACE_Attention/distill/RACE2_distillation_explained.pdf")

# ======================================================================= PAGE 1
fig, ax = newpage(pdf)
box(ax, 0.0, 0.93, 1.0, 0.07, INK)
ax.text(0.06, 0.974, "RACE 2.0 — Distilling an LLM into a Linear-Time Hybrid",
        transform=ax.transAxes, fontsize=15, color="white", weight="bold",
        ha="left", va="top", wrap=False, zorder=3)
txt(ax, 0.06, 0.943, "What the distillation does and what objective we optimize", size=11, color="#C7D2FE")

# exec framing
box(ax, 0.06, 0.79, 0.88, 0.11, LGREY, ec="#CBD5E1", lw=1)
txt(ax, 0.085, 0.875, "The one-line idea", size=12.5, color=INK, weight="bold")
txt(ax, 0.085, 0.845,
    "Standard transformers cost O(T²) in sequence length — they get slow and memory-hungry on long\n"
    "inputs, and pay that tax on every request. We take a model that already works (Llama-3.2-3B) and,\n"
    "with a one-time training step, convert it into a hybrid that runs in linear time O(T) at long context\n"
    "— faster for life — while teaching it to keep the original model’s accuracy.", size=10.5, color=GREY)

txt(ax, 0.06, 0.74, "The architecture: teacher → student", size=13.5, color=INK, weight="bold")

# teacher stack
tx, ty, tw = 0.10, 0.40, 0.20
box(ax, tx, ty, tw, 0.28, LBLUE, ec=BLUE, lw=1.6)
txt(ax, tx+tw/2, 0.70, "TEACHER", size=11, color=BLUE, weight="bold", ha="center")
txt(ax, tx+tw/2, 0.685, "Llama-3.2-3B (frozen)", size=8.5, color=GREY, ha="center")
for i in range(7):
    yy = ty + 0.012 + i*0.038
    box(ax, tx+0.025, yy, tw-0.05, 0.028, BLUE, alpha=0.55)
txt(ax, tx+tw/2, ty-0.022, "28 softmax-attention layers", size=9, color=INK, ha="center", weight="bold")
txt(ax, tx+tw/2, ty-0.045, "quadratic  O(T²)", size=8.5, color=BLUE, ha="center")

# arrow + label
arrow(ax, tx+tw+0.02, 0.54, tx+tw+0.16, 0.54, color=INK, lw=2.6, ms=20)
txt(ax, tx+tw+0.09, 0.575, "distill", size=11, color=INK, weight="bold", ha="center")
txt(ax, tx+tw+0.09, 0.50, "(one-time)", size=8.5, color=GREY, ha="center")

# student stack (alternating)
sx = tx+tw+0.18
box(ax, sx, ty, tw, 0.28, LTEAL, ec=TEAL, lw=1.6)
txt(ax, sx+tw/2, 0.70, "STUDENT", size=11, color=TEAL, weight="bold", ha="center")
txt(ax, sx+tw/2, 0.685, "RACE-hybrid (trained)", size=8.5, color=GREY, ha="center")
for i in range(7):
    yy = ty + 0.012 + i*0.038
    c = ORANGE if i % 2 == 0 else BLUE
    box(ax, sx+0.025, yy, tw-0.05, 0.028, c, alpha=0.7)
txt(ax, sx+tw/2, ty-0.022, "14 softmax  +  14 RACE", size=9, color=INK, ha="center", weight="bold")
txt(ax, sx+tw/2, ty-0.045, "linear  O(T)  in the RACE layers", size=8.5, color=TEAL, ha="center")

# legend
lx = sx+tw+0.04
box(ax, lx, 0.55, 0.018, 0.018, BLUE, alpha=0.7); txt(ax, lx+0.03, 0.568, "softmax (kept)", size=8.5, color=GREY)
box(ax, lx, 0.50, 0.018, 0.018, ORANGE, alpha=0.7); txt(ax, lx+0.03, 0.518, "RACE (linear,", size=8.5, color=GREY)
txt(ax, lx+0.03, 0.498, "replaces softmax)", size=8.5, color=GREY)

box(ax, 0.06, 0.10, 0.88, 0.22, LTEAL, ec=TEAL, lw=1)
txt(ax, 0.085, 0.30, "What is RACE?", size=12.5, color=TEAL, weight="bold")
txt(ax, 0.085, 0.27,
    "RACE is a linear-time replacement for softmax attention: instead of comparing every token to every\n"
    "other token (the O(T²) cost), it hashes tokens into a small number of buckets and aggregates within\n"
    "them via a running prefix-sum — O(T) time and O(1) memory per token at decode. We keep half the\n"
    "original softmax layers because they preserve exact long-range retrieval (finding a specific fact in a\n"
    "long document) that the linear approximation alone struggles with. The result is a tunable\n"
    "speed ↔ accuracy trade-off: more RACE layers → faster; more softmax layers → stronger retrieval.", size=10.3, color=GREY)
txt(ax, 0.06, 0.055, "RACE 2.0  ·  Distillation & Optimization Objective", size=8.5, color="#94A3B8")
txt(ax, 0.94, 0.055, "1 / 3", size=8.5, color="#94A3B8", ha="right")
pdf.savefig(fig); fig.savefig("/tmp/race_pdf_p1.png", dpi=120); plt.close(fig)

# ======================================================================= PAGE 2
fig, ax = newpage(pdf)
box(ax, 0.0, 0.93, 1.0, 0.07, INK)
txt(ax, 0.06, 0.972, "The Optimization Objective", size=17, color="white", weight="bold")
txt(ax, 0.06, 0.945, "What “distillation” means precisely: matching the teacher at three levels", size=11, color="#C7D2FE")

txt(ax, 0.06, 0.90,
    "Both models read the same input text. The teacher is frozen; the student is trained so its behaviour\n"
    "matches the teacher’s. We minimise a weighted sum of three losses:", size=10.5, color=GREY)

# total loss box
box(ax, 0.10, 0.78, 0.80, 0.065, "#0B1220")
ax.text(0.5, 0.8125, r"$\mathcal{L}\;=\;0.1\;\mathcal{L}_{\mathrm{hidden}}\;+\;1.0\;\mathcal{L}_{\mathrm{KL}}\;+\;0.5\;\mathcal{L}_{\mathrm{CE}}$",
        transform=ax.transAxes, fontsize=19, color="white", ha="center", va="center", zorder=4)

def term(ax, y, accent, light, name, weight_lbl, eq, plain):
    box(ax, 0.06, y, 0.88, 0.155, light, ec=accent, lw=1.4)
    box(ax, 0.06, y+0.123, 0.88, 0.032, accent)
    txt(ax, 0.085, y+0.150, name, size=12, color="white", weight="bold")
    txt(ax, 0.915, y+0.150, weight_lbl, size=10.5, color="white", weight="bold", ha="right")
    ax.text(0.10, y+0.088, eq, transform=ax.transAxes, fontsize=14.5, color=INK, ha="left", va="center", zorder=4)
    txt(ax, 0.085, y+0.052, plain, size=9.7, color=GREY)

term(ax, 0.585, BLUE, LBLUE, "1.  Logit distillation  —  “dark knowledge”", "weight 1.0  (primary)",
     r"$\mathcal{L}_{\mathrm{KL}} = T^2\,\mathrm{KL}\left(\mathrm{softmax}(z^{t}/T)\,\|\,\mathrm{softmax}(z^{s}/T)\right)$",
     "The student matches the teacher’s full probability distribution over all ~128,000 possible next tokens\n"
     "— not just the correct answer, but how the teacher ranks every alternative. These “soft” targets carry\n"
     "far more signal than the single right token. (z = logits; T = temperature, see note below.)")

term(ax, 0.385, GREEN, LGRN, "2.  Feature / representation matching", "weight 0.1",
     r"$\mathcal{L}_{\mathrm{hidden}} = \frac{1}{|R|}\sum_{\ell\in R}\left\|\,h^{s}_{\ell}-h^{t}_{\ell}\,\right\|^{2}$",
     "At each replaced layer ℓ, the student’s internal representation h is pulled toward the teacher’s, so the\n"
     "new linear attention produces teacher-like activations — not just teacher-like final answers. (R = the\n"
     "set of replaced layers.)")

term(ax, 0.185, ORANGE, LOR, "3.  Language-model loss  (hard labels)", "weight 0.5",
     r"$\mathcal{L}_{\mathrm{CE}} = -\sum_{t}\log p^{s}\left(x_{t+1}\mid x_{\leq t}\right)$",
     "Standard next-token prediction against the ground-truth text. Keeps the student a fluent language\n"
     "model and directly shapes the token it will actually generate.")

box(ax, 0.06, 0.075, 0.88, 0.085, LGREY, ec="#CBD5E1", lw=1)
txt(ax, 0.085, 0.150, "Two important details", size=10.5, color=INK, weight="bold")
txt(ax, 0.085, 0.122,
    "•  Temperature T (“dark knowledge”): T>1 softens the teacher’s distribution to emphasise informative\n"
    "   low-probability tokens. We currently use T = 1 (match the raw probabilities); the T² factor keeps the\n"
    "   gradient scale stable if we raise it.    •  The teacher is frozen — gradients update only the student’s\n"
    "   new RACE attention layers (and, in the latest stage, its feed-forward MLPs).", size=9.3, color=GREY)
txt(ax, 0.06, 0.045, "RACE 2.0  ·  Distillation & Optimization Objective", size=8.5, color="#94A3B8")
txt(ax, 0.94, 0.045, "2 / 3", size=8.5, color="#94A3B8", ha="right")
pdf.savefig(fig); fig.savefig("/tmp/race_pdf_p2.png", dpi=120); plt.close(fig)

# ======================================================================= PAGE 3
fig, ax = newpage(pdf)
box(ax, 0.0, 0.93, 1.0, 0.07, INK)
txt(ax, 0.06, 0.972, "How a Training Step Works", size=17, color="white", weight="bold")
txt(ax, 0.06, 0.945, "Same input → two models → compare outputs → update only the student", size=11, color="#C7D2FE")

# input
box(ax, 0.40, 0.85, 0.20, 0.045, LGREY, ec="#CBD5E1", lw=1)
txt(ax, 0.50, 0.8725, "input tokens", size=10, color=INK, ha="center", va="center", weight="bold")
# teacher + student boxes
box(ax, 0.10, 0.66, 0.34, 0.11, LBLUE, ec=BLUE, lw=1.6)
txt(ax, 0.27, 0.745, "TEACHER  (frozen)", size=11, color=BLUE, ha="center", weight="bold")
txt(ax, 0.27, 0.715, "Llama-3.2-3B, all softmax", size=9, color=GREY, ha="center")
txt(ax, 0.27, 0.688, "→ hidden states + logits  (targets)", size=9, color=INK, ha="center")
box(ax, 0.56, 0.66, 0.34, 0.11, LTEAL, ec=TEAL, lw=1.6)
txt(ax, 0.73, 0.745, "STUDENT  (trained)", size=11, color=TEAL, ha="center", weight="bold")
txt(ax, 0.73, 0.715, "RACE-hybrid", size=9, color=GREY, ha="center")
txt(ax, 0.73, 0.688, "→ hidden states + logits", size=9, color=INK, ha="center")
arrow(ax, 0.46, 0.85, 0.27, 0.775, color=BLUE, lw=2); arrow(ax, 0.54, 0.85, 0.73, 0.775, color=TEAL, lw=2)
# compare box
box(ax, 0.22, 0.50, 0.56, 0.10, "#0B1220")
txt(ax, 0.50, 0.575, "COMPARE  (the 3-term objective)", size=11, color="white", ha="center", weight="bold")
txt(ax, 0.50, 0.545, "logit-KL  +  hidden-MSE  +  cross-entropy", size=10, color="#C7D2FE", ha="center")
txt(ax, 0.50, 0.520, "(teacher outputs are fixed targets)", size=8.5, color="#94A3B8", ha="center")
arrow(ax, 0.27, 0.66, 0.40, 0.60, color=BLUE, lw=2); arrow(ax, 0.73, 0.66, 0.60, 0.60, color=TEAL, lw=2)
# update
box(ax, 0.34, 0.40, 0.32, 0.05, LOR, ec=ORANGE, lw=1.4)
txt(ax, 0.50, 0.425, "update ONLY the student", size=10.5, color=ORANGE, ha="center", va="center", weight="bold")
arrow(ax, 0.50, 0.50, 0.50, 0.452, color=INK, lw=2)
arrow(ax, 0.66, 0.425, 0.80, 0.425, color=GREY, lw=1.6, style="-|>")
txt(ax, 0.815, 0.425, "repeat", size=9, color=GREY, va="center")
arrow(ax, 0.84, 0.44, 0.84, 0.71, color=GREY, lw=1.2, style="-|>")

# payoff
box(ax, 0.06, 0.12, 0.88, 0.22, LTEAL, ec=TEAL, lw=1)
txt(ax, 0.085, 0.32, "Why this pays off", size=12.5, color=TEAL, weight="bold")
txt(ax, 0.085, 0.292,
    "The conversion is a one-time cost (~0.7–1B training tokens, < 0.01% of pre-training). After that the\n"
    "hybrid is permanently cheaper at long context:", size=10.3, color=GREY)
txt(ax, 0.105, 0.235,
    "•  Decode is faster at every length and the gap grows with context — up to ~1.9× faster per token at\n"
    "   64K (batch 8), and it uses ~30% less memory (so it serves longer / bigger batches before running out).\n"
    "•  Prefill becomes faster beyond ~32K tokens.\n"
    "•  Accuracy is recovered by the distillation: general knowledge (MMLU) and long-context retrieval\n"
    "   (RULER) both climb back toward the teacher as training proceeds.", size=10.0, color=GREY)
txt(ax, 0.085, 0.135, "Net: “train once, run fast for life” — without giving up the model’s accuracy.",
    size=10.5, color=INK, weight="bold")
txt(ax, 0.06, 0.045, "RACE 2.0  ·  Distillation & Optimization Objective", size=8.5, color="#94A3B8")
txt(ax, 0.94, 0.045, "3 / 3", size=8.5, color="#94A3B8", ha="right")
pdf.savefig(fig); fig.savefig("/tmp/race_pdf_p3.png", dpi=120); plt.close(fig)

pdf.close()
print("wrote /scratch/sj157/RACE_Attention/distill/RACE2_distillation_explained.pdf")
