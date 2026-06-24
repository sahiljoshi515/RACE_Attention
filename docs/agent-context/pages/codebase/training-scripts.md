# Training & example scripts (`misc/`, `notebooks/`)

End-to-end task scripts demonstrating RACE as a drop-in attention. Map directly to the paper's
experiment suites (`paper/07-experiments`).

## `misc/` scripts

| File | Task | Paper table |
| --- | --- | --- |
| `misc/race.py` | RACE modules (`BatchedACE`, `RACEAttention`, `RACEBlock`) - imported by the others | Algorithm 1/2 |
| `misc/gpt.py` | GPT model + reference softmax `MultiHeadAttention`/`TransformerBlock` | baseline |
| `misc/classification.py` | Text classification (IMDB) | Table 2, appendix |
| `misc/mlm.py` | Masked LM, BERT-style (Tiny Stories) | Table 2, appendix |
| `misc/lm.py` | Autoregressive LM (WikiText-103) - causal | Tables 4-5 |
| `misc/vit.py` | Vision Transformer (non-causal RACE) | Table 3 |
| `misc/food-101.py` | Image classification @16K | Table 3 |
| `misc/arxiv_64K.py` | Long-context Arxiv classification @64K | Table 1 |

## `notebooks/` (runnable quickstarts, ~5-10 min)

- `ClassificationTask.ipynb` (IMDB @512), `LanguageModelling.ipynb` (WikiText-103 @128),
  `MaskedLanguageModelling.ipynb` (Tiny Stories @512), `VisionTask.ipynb` (FashionMNIST @784).

All show RACE as a drop-in replacement for softmax attention with a training loop and baseline
comparison. Start here to see the API in action before touching kernels.

---
Source: misc/*.py, notebooks/*.ipynb file inventory + README.md. Verified against HEAD c620cdc.
