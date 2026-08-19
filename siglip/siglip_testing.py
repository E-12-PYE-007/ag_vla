"""
siglip_testing.py

Captum-based diagnostic maps for the SigLIP SO400M vision encoder extracted from
OmniVLA (AsyncVLA_release checkpoint).

Two maps are computed and overlaid on each image in IMAGE_DIR:

  1. Self-Attention Map  — captures last-block attention weights via Captum
                           LayerActivation on the attention dropout layer.
                           Shows which image patches collectively attract the most
                           attention (column-sum saliency, mean over heads).

  2. Cross-Modal Map     — uses Captum LayerIntegratedGradients at the last ViT
                           block to attribute the text–image cosine similarity score
                           back to each patch position.  Shows which image regions
                           drive the model's agreement with the language instruction.

The language instruction used is printed in the figure title so context is always clear.
Results are saved to OUT_DIR as {image_stem}_siglip_maps.png.

Usage:
    python siglip_testing.py
    python siglip_testing.py --text "navigate to the fence"

Requires:
    pip install captum
    prismatic package on Python path (from the AsyncVLA repo)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import LayerActivation, LayerIntegratedGradients
from captum.attr import visualization as viz
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# ── Constants ──────────────────────────────────────────────────────────────────

ASYNCVLA_RELEASE_PATH = "/home/vla-cap/AsyncVLA/AsyncVLA_release"
SIGLIP_HF_MODEL_ID = "google/siglip-so400m-patch14-224"

IMAGE_SIZE = 224
PATCH_SIZE = 14
GRID_H = GRID_W = IMAGE_SIZE // PATCH_SIZE   # 16 × 16 = 256 patches

DEVICE = "cuda"
IG_STEPS = 50

# Map each language instruction to the folder containing its test images.
# Add or remove entries as needed.
INSTRUCTION_DIRS = {
    "navigate to the fence": "/home/vla-cap/data/siglip_test/navigate_to_fence",
    "go to the gate":        "/home/vla-cap/data/siglip_test/go_to_gate",
}

# All outputs are saved here as {instruction_slug}/{image_stem}_siglip_maps.png
OUT_DIR = "/home/vla-cap/data/siglip_results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# SigLIP preprocessing: resize-naive 224×224, bicubic, mean/std = 0.5
SIGLIP_IMAGE_PROCESSOR = PrismaticImageProcessor(
    use_fused_vision_backbone=False,
    image_resize_strategy="resize-naive",
    input_sizes=[(3, IMAGE_SIZE, IMAGE_SIZE)],
    interpolations=["bicubic"],
    means=[(0.5, 0.5, 0.5)],
    stds=[(0.5, 0.5, 0.5)],
)


# ── HF class registration ──────────────────────────────────────────────────────

def _register_hf_classes() -> None:
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    except ValueError:
        pass


# ── Model loading ──────────────────────────────────────────────────────────────

def load_siglip(device: torch.device):
    """
    Load OmniVLA from AsyncVLA_release, extract the SigLIP SO400M featurizer,
    and free the rest of the VLA from memory.

    In PrismaticVisionBackbone (HF wrapper):
        featurizer       → DINOv2 ViT-L (timm_model_ids[0])
        fused_featurizer → SigLIP SO400M (timm_model_ids[1])
    """
    _register_hf_classes()
    print(f"Loading OmniVLA from {ASYNCVLA_RELEASE_PATH} ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        ASYNCVLA_RELEASE_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    siglip = vla.vision_backbone.fused_featurizer.float().eval().to(device)
    n = sum(p.numel() for p in siglip.parameters()) / 1e6
    print(f"  SigLIP SO400M: {n:.0f} M params  (patch14, {GRID_H}×{GRID_W} grid)")
    del vla
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return siglip


def get_text_embedding(text: str, device: torch.device) -> torch.Tensor:
    """Return a normalised (1, D) text embedding from the HF SigLIP SO400M text encoder."""
    print(f"Loading text encoder ({SIGLIP_HF_MODEL_ID}) ...")
    text_model = AutoModel.from_pretrained(SIGLIP_HF_MODEL_ID).to(device).eval()
    text_proc  = AutoProcessor.from_pretrained(SIGLIP_HF_MODEL_ID)
    inputs = text_proc(text=[text], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        emb = text_model.get_text_features(**inputs).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    del text_model
    return emb   # (1, D)


# ── Captum wrapper ─────────────────────────────────────────────────────────────

class SiglipSimilarityWrapper(torch.nn.Module):
    """
    Thin wrapper so Captum sees a single-input → single-scalar model.

    forward() calls forward_features() directly on the TIMM model, bypassing the
    OmniVLA monkey-patch that returns second-to-last layer features.  This ensures
    all ViT blocks (including blocks[-1]) are traversed so Captum's hooks land
    on the right layer.

    Returns the cosine similarity between the mean-pooled patch embedding and the
    pre-computed text embedding.  Shape: (B,) — a scalar per sample.
    """

    def __init__(self, siglip, text_emb: torch.Tensor) -> None:
        super().__init__()
        self.siglip = siglip
        self.register_buffer("text_emb", text_emb)  # (1, D)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.siglip.forward_features(pixel_values)   # (B, 256, D)
        x = x.mean(dim=1)                                # (B, D) — mean pool, no CLS
        x = x / x.norm(dim=-1, keepdim=True)
        return (x @ self.text_emb.T).squeeze(-1)         # (B,)


# ── Diagnostic maps ────────────────────────────────────────────────────────────

def compute_self_attention_map(
    wrapper: SiglipSimilarityWrapper,
    pixel_values: torch.Tensor,
) -> np.ndarray:
    """
    Use Captum LayerActivation to capture the softmax attention weights from the
    last ViT block's attention dropout layer.

    attn_drop receives the (B, heads, N, N) softmax attention matrix as its input
    and (in eval mode) passes it through unchanged.  LayerActivation captures this
    output directly without computing any gradient.

    Aggregation: mean over attention heads → column-sum over query dimension
    → gives (N,) "total attention received" saliency per patch
    → reshaped to (GRID_H, GRID_W) and min-max normalised.
    """
    la = LayerActivation(wrapper, wrapper.siglip.blocks[-1].attn.attn_drop)

    with torch.no_grad():
        activation = la.attribute(pixel_values)   # (B, heads, N, N)

    attn = activation[0].detach().cpu().float()   # (heads, N, N)
    saliency = attn.mean(0).sum(0).numpy()        # (N,) column-sum of mean-head attn

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.reshape(GRID_H, GRID_W)       # (16, 16)


def compute_cross_modal_map(
    wrapper: SiglipSimilarityWrapper,
    pixel_values: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Use Captum LayerIntegratedGradients at the last ViT block to attribute the
    text–image similarity score back to each patch position.

    The baseline is a tensor of zeros in normalised pixel space (a neutral grey image).
    IG integrates the gradient of the similarity score w.r.t. the last block's output
    along the straight-line path from baseline to input.

    Returned attribution shape: (B, N, D).  We take the L2 norm over the feature
    dimension D to collapse to (B, N), then reshape to (GRID_H, GRID_W).
    """
    lig = LayerIntegratedGradients(wrapper, wrapper.siglip.blocks[-1])

    baseline = torch.zeros_like(pixel_values)

    attrs = lig.attribute(
        pixel_values,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=1,
    )   # (B, N, D)

    # L2 norm over feature dimension → per-patch importance (B, N)
    importance = attrs[0].detach().cpu().float().norm(dim=-1).numpy()   # (N,)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance.reshape(GRID_H, GRID_W)   # (16, 16)


# ── Upsampling ─────────────────────────────────────────────────────────────────

def upsample_map(heatmap: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear upsample a (H, W) float32 heatmap to (target_h, target_w)."""
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()   # (target_h, target_w)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(
    img_arr: np.ndarray,
    attn_map: np.ndarray,
    cross_map: np.ndarray,
    text: str,
    out_path: str,
) -> None:
    """
    Three-panel figure using Captum's visualize_image_attr for the two overlays.
    Captum handles colour mapping, alpha blending, and colourbar rendering.
    """
    h, w = img_arr.shape[:2]
    attn_up  = upsample_map(attn_map,  h, w)
    cross_up = upsample_map(cross_map, h, w)

    # Captum expects attributions as (H, W, 1) or (H, W, C) numpy arrays
    attn_attr  = attn_up[..., np.newaxis]    # (H, W, 1)
    cross_attr = cross_up[..., np.newaxis]   # (H, W, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 0: original image
    axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 1: self-attention overlay via Captum
    viz.visualize_image_attr(
        attn_attr,
        img_arr,
        method="blended_heat_map",
        sign="positive",
        cmap="viridis",
        alpha_overlay=0.55,
        show_colorbar=True,
        title="Self-Attention\n(last ViT block · mean head · patch saliency)",
        plt_fig_axis=(fig, axes[1]),
        use_pyplot=False,
    )
    axes[1].title.set_fontsize(9)

    # Panel 2: cross-modal IG overlay via Captum
    viz.visualize_image_attr(
        cross_attr,
        img_arr,
        method="blended_heat_map",
        sign="positive",
        cmap="plasma",
        alpha_overlay=0.55,
        show_colorbar=True,
        title="Cross-Modal Attribution\n(Integrated Gradients · patch → text similarity)",
        plt_fig_axis=(fig, axes[2]),
        use_pyplot=False,
    )
    axes[2].title.set_fontsize(9)

    fig.suptitle(
        f'Language instruction:  "{text}"\n'
        f"SigLIP SO400M  ·  OmniVLA AsyncVLA_release weights",
        fontsize=10,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def process_image(
    image: Image.Image,
    text: str,
    wrapper: SiglipSimilarityWrapper,
    dev: torch.device,
    out_path: str,
) -> None:
    """Run both diagnostic maps for a single image and save the figure."""
    rgb          = image.convert("RGB")
    pixel_values = SIGLIP_IMAGE_PROCESSOR.apply_transform(rgb).unsqueeze(0).to(dev)
    img_arr      = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

    with torch.no_grad():
        score = wrapper(pixel_values).item()
    print(f"  similarity: {score:.4f}")

    attn_map  = compute_self_attention_map(wrapper, pixel_values)
    cross_map = compute_cross_modal_map(wrapper, pixel_values, n_steps=IG_STEPS)

    plot_results(img_arr, attn_map, cross_map, text, out_path)


def main() -> None:
    dev = torch.device(DEVICE)

    # Load SigLIP once; reuse across all instructions and images
    siglip = load_siglip(dev)

    for text, img_dir in INSTRUCTION_DIRS.items():
        img_dir = Path(img_dir)
        if not img_dir.exists():
            print(f"[skip] directory not found: {img_dir}")
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"[skip] no images found in {img_dir}")
            continue

        print(f'\n── "{text}" ({len(images)} images) ──')

        # Text embedding is the same for all images under this instruction
        text_emb = get_text_embedding(text, dev)
        wrapper  = SiglipSimilarityWrapper(siglip, text_emb).to(dev)

        # Output subfolder: OUT_DIR / instruction slug
        slug    = text.replace(" ", "_")
        out_dir = Path(OUT_DIR) / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            print(f"  {img_path.name}", end="  ")
            out_path = out_dir / f"{img_path.stem}_siglip_maps.png"
            image = Image.open(img_path).convert("RGB")
            process_image(image, text, wrapper, dev, str(out_path))
            print(f"→ {out_path.name}")


if __name__ == "__main__":
    main()
