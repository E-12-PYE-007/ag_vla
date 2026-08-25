# SigLIP saliency maps for AsyncVLA / OmniVLA

`siglip_maps.py` renders text-conditioned saliency maps for the **SigLIP SO400M**
vision tower that AsyncVLA (OmniVLA, `AsyncVLA_release`) uses as its image encoder.

For each `(instruction, image)` pair it produces a 3-panel figure:

```
Original | ℱ₆ Grad-CAM | ℱ₂ grad-weighted attention rollout (Chefer)
```

Both attribution panels target SigLIP's **text–image cosine similarity** — i.e.
"which image regions drive agreement with the instruction."

## Run

```bash
# in the asyncvla conda env (has prismatic, torch 2.2.0, captum)
cd siglip
python siglip_maps.py
```

It loads OmniVLA's fine-tuned SigLIP tower + a stock SigLIP head, prints the A/B/C
similarity sanity table, and writes figures to `results_async_stockhead/<instruction>/`.

Edit `INSTRUCTION_DIRS` at the top of the script to point at your image folders.

---

## SigLIP SO400M architecture (and where each method taps in)

```
   image 224×224×3
        │  patch embed (14×14 patches) + positional embedding
        ▼
   256 patch tokens × 1152-d          ← NO [CLS] token, no registers
        │
        │  block 0   ── self-attn A₀  (16 heads, 256×256) ┐
        │  block 1   ── self-attn A₁                      │
        │   …                                             │  ← ℱ₂ Chefer taps the
        │  block 25  (LoRA-fine-tuned in OmniVLA)         │    attention A_l of
        │  block 26  (stock) ── self-attn A₂₆ ────────────┘    EVERY block
        ▼
   ═══ block-26 OUTPUT: 256 × 1152 ═══     ← ℱ₆ Grad-CAM taps HERE (one layer)
        │  final LayerNorm
        ▼
   attention-pool head: 1 learned query ── cross-attn A_pool (1×256) ── over 256 patches
        │                                   ↑ ℱ₂ Chefer reads out here;
        ▼                                     ℱ₆ Grad-CAM's gradient flows through here
   image embedding (1152)
        │                text "Follow the perimeter…" → text tower → text emb (1152)
        └──────────── cosine similarity  (scalar both methods differentiate) ──────────┘
```

Key facts: **1152-d width, 27 blocks, 16 heads, 256 patch tokens, no CLS, a learned
attention-pool head, and a separate text tower.** It is a **dual-encoder** — the
vision tower never sees the text; the only coupling is the final cosine, so the
instruction's entire influence enters through `∂(cosine)`.

---

## ℱ₆ Grad-CAM — attributes to token *features* at one depth

- **Taps:** the output of **block 26** — the `256 × 1152` patch-token tensor in the
  residual stream (ℱ₆ = post-FFN residual output). One layer.
- **Architecture analogy:** treat the **1152 feature dims as "channels"** and the
  **256 patches as the "spatial grid."** SigLIP having no CLS is convenient — the
  whole tensor is already a `16×16×1152` feature map.
- **Computation:** forward to the cosine scalar → backprop → at block-26 output grab
  activations `A` and gradients `G`:

  ```
  αc      = mean_n  ∂sim/∂A[n, c]           # per-channel importance
  cam[n]  = ReLU( Σc αc · A[n, c] )          # per-patch saliency
  ```
  reshape 256 → 16×16, min-max normalize, upsample.
- **Means:** "which patch's *content* most drives the match." Content contribution.
- **Taxonomy / lineage:** ℱ₆ token-feature Grad-CAM = the **CLIP-ES** family
  (dual-encoder, vision-encoder token features, image–text similarity target).

## ℱ₂ Chefer rollout — attributes to attention *routing* across all depths

- **Taps:** the self-attention matrix `A_l` (256×256) **inside every one of the 27
  blocks**, plus the pool head's cross-attention `A_pool` (1×256).
- **Architecture:** the blocks repeatedly **mix the 256 patch tokens among
  themselves**; the pool head **mixes 256 patches into the 1 image-embedding query.**
- **Computation:**

  ```
  per block:  C_l = ReLU( A_l ⊙ ∂sim/∂A_l ), mean over heads      → (256, 256)
  rollout:    Â_l = rownorm( C_l + I ),   R = Â_l @ R  (R starts at I)
  readout:    p   = ReLU( A_pool ⊙ ∂sim/∂A_pool ), mean heads     → (1, 256)
              saliency = p @ R                                     → per patch
  ```
  - `⊙` keeps an attention edge only if it is active **and** raises the similarity.
  - `+ I` models the **residual/skip connection** (a token also carries its own
    representation forward, not only what attention routes to it).
  - **No CLS** in SigLIP → we cannot take a CLS row of `R`; instead we **read out
    through the attention-pool head** (the query that forms the image embedding).
    This is the key SigLIP-specific adaptation.
- **Means:** "which input patches the instruction-relevant *information flow*
  concentrates on, after 27 rounds of token mixing + final pooling." Routing
  contribution.
- **Taxonomy / lineage:** **Chefer et al.** mechanics (grad-weighted attention
  rollout) with a **CLIP-ES** target (image–text similarity).

## Side by side

| | ℱ₆ Grad-CAM | ℱ₂ Chefer rollout |
|---|---|---|
| Taps into | block-26 **features** (residual stream) | **attention** of all 27 blocks + pool head |
| Depth | one layer | every layer |
| Explains | patch **content** contribution | patch **routing** contribution |
| Needs CLS? | no (256 patches = grid) | no — **reads out via the pool head** |
| Text enters via | `∂(cosine)` only | `∂(cosine)` only |

Both hinge on two SigLIP specifics: **no CLS** (clean grid for Grad-CAM; pool-head
readout for Chefer) and **the attention-pool head as the 256→1 bottleneck** (Grad-CAM's
gradient passes through it; Chefer uses it as the readout query).

---

## Attention sinks

An **attention sink** is a token that absorbs a large share of attention **regardless
of image content** — softmax must sum to 1, so when a head has nothing useful to
attend to it dumps the leftover mass onto a few fixed, low-information tokens (a
learned no-op). In ViTs these show up as high-norm patch tokens (cf. the DINOv2
"registers" work). Symptom here: a fixed bright hotspot at the same grid position
across *all* images. **A sink is high-attention, low-importance** — which is exactly
why "attention ≠ explanation" and why gradient methods exist.

Gradient weighting (both methods) **suppresses** sinks — a sink rarely moves the
similarity, so `∂sim` there is small — but SigLIP's sinks are strong enough that some
survive in the ℱ₂ rollout, which is why it stays a bit spottier than the ℱ₆ Grad-CAM.

---

## Caveats (important for interpretation / the writeup)

- **SigLIP-native probe, not VLA grounding.** In AsyncVLA the instruction only enters
  the **LLM**; SigLIP is a pure visual featurizer (block-25 patch tokens → projector →
  LLM). These text-similarity maps use SigLIP's own contrastive head + text tower,
  which the VLA never runs. They measure SigLIP-style grounding of the features, not
  how the policy uses language.
- **Fine-tuning split.** OmniVLA LoRA-fine-tuned blocks 0–25; block 26 + attention-pool
  head + final norm stayed at stock (they get no gradient because the VLA taps the
  2nd-to-last block). So ℱ₆ Grad-CAM reads stock block-26 features sitting on fine-tuned
  0–25 features, while the ℱ₂ rollout spans both.
- **Raw cosines are tiny (~0.05–0.12).** Normal for SigLIP — it applies
  `logit_scale ≈ 111`, `bias ≈ −16.5` before its sigmoid. Treat the maps as relative,
  not calibrated.
- **Method choice matters (arXiv 2608.05258).** "Grad-CAM on a ViT" is ambiguous; this
  script fixes explicit choices — feature location (ℱ₆ block-26 / ℱ₂ all-block
  attention), gradient target (image–text cosine), token handling (no CLS), head
  aggregation (mean), and 16×16 reshape.
