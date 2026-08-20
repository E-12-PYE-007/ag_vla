"""
siglip_head_ablation.py

Controlled ablation: hold the SigLIP attention-pool HEAD fixed, swap the TOWER.

Three configs, same text embedding + same preprocessing:
  A) stock  tower + stock head   (pure stock SigLIP)
  B) async  tower + stock head   (AsyncVLA fine-tuned features, stock grounding head)
  C) async  tower + async head   (what siglip_omnivla_maps.py already does)

Purpose:
  * B is the requested test: "use the async SigLIP but take the attention head of
    stock SigLIP."
  * B vs C verifies the async head is really identical to stock (it got no gradient
    during finetune), i.e. B == C numerically.
  * A vs B isolates the effect of the LoRA-finetuned vision FEATURES on text
    grounding, with the grounding head held constant.

Grad-CAM maps for config B are written to results_async_stockhead/.

Usage:
    python siglip_head_ablation.py
"""

from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from siglip_testing import (
    INSTRUCTION_DIRS, IMAGE_EXTENSIONS, IMAGE_SIZE, IG_STEPS,
    SIGLIP_IMAGE_PROCESSOR, DEVICE,
    load_siglip, get_text_embedding, compute_cross_modal_map,
)
from siglip_stock_maps import compute_gradcam_map, plot_gradcam_vs_ig

STOCK_TIMM_ID = "vit_so400m_patch14_siglip_224"
OUT_DIR = str(Path(__file__).resolve().parent / "results_async_stockhead")


class TowerHeadWrapper(torch.nn.Module):
    """similarity = cos( head(tower.forward_features(x)) , text ).  Head is swappable."""
    def __init__(self, tower, head, text_emb):
        super().__init__()
        self.siglip = tower          # named .siglip so compute_gradcam_map can hook .blocks
        self.head = head
        self.register_buffer("text_emb", text_emb)

    def forward(self, px):
        feats = self.siglip.forward_features(px)   # (B,256,D) — all blocks + final norm
        pooled = self.head(feats)                  # (B,D) via attention-pool head
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return (pooled @ self.text_emb.T).squeeze(-1)


def sim(tower, head, text_emb, px):
    with torch.no_grad():
        feats = tower.forward_features(px)
        pooled = head(feats)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return (pooled @ text_emb.T).squeeze().item()


def main():
    dev = torch.device(DEVICE)
    async_tower = load_siglip(dev)                                   # AsyncVLA fused_featurizer
    stock = timm.create_model(STOCK_TIMM_ID, pretrained=True, num_classes=0).float().eval().to(dev)
    stock_head, async_head = stock.attn_pool, async_tower.attn_pool

    # quick head-identity check
    d = sum((pa.data - ps.data).norm().item()
            for pa, ps in zip(async_head.parameters(), stock_head.parameters()))
    tot = sum(ps.data.norm().item() for ps in stock_head.parameters())
    print(f"\nasync head vs stock head: summed ||Δ|| = {d:.4f}  (rel {d/tot:.5f})  -> "
          f"{'IDENTICAL' if d/tot < 1e-2 else 'DIFFERENT'}")

    print(f"\n{'instruction':34s} {'A stock+stock':>14s} {'B async+stock':>14s} {'C async+async':>14s}")
    for text, dpath in INSTRUCTION_DIRS.items():
        imgs = sorted(p for p in Path(dpath).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not imgs:
            continue
        te = get_text_embedding(text, dev)
        px = SIGLIP_IMAGE_PROCESSOR.apply_transform(
            Image.open(imgs[0]).convert("RGB")).unsqueeze(0).to(dev)
        a = sim(stock,       stock_head, te, px)
        b = sim(async_tower, stock_head, te, px)
        c = sim(async_tower, async_head, te, px)
        print(f"{text:34s} {a:>+14.4f} {b:>+14.4f} {c:>+14.4f}")

        # Grad-CAM for config B (async features, stock head), for the record
        wrapper = TowerHeadWrapper(async_tower, stock_head, te).to(dev)
        rgb = Image.open(imgs[0]).convert("RGB")
        img_arr = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype("float32") / 255.0
        gradcam = compute_gradcam_map(wrapper, px)
        ig      = compute_cross_modal_map(wrapper, px, n_steps=IG_STEPS)
        out = Path(OUT_DIR) / text.replace(" ", "_")
        out.mkdir(parents=True, exist_ok=True)
        plot_gradcam_vs_ig(img_arr, gradcam, ig, text,
                           str(out / f"{imgs[0].stem}_siglip_maps.png"),
                           model_label="AsyncVLA SigLIP features + STOCK attn-pool head")


if __name__ == "__main__":
    main()
