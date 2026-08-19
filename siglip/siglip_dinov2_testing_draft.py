"""
test_siglip.py

Grad-CAM analysis of the SigLIP vision encoder extracted from the OmniVLA
AsyncVLA_release checkpoint.

The model loaded here is the full OpenVLAForActionPrediction_MMNv1; we then
pull out vla.vision_backbone.siglip_featurizer (TIMM vit_so400m_patch14_siglip_224)
and discard the rest.  This gives us the TRAINED SigLIP weights rather than the
generic pretrained ones.

For the text side we use the matching standalone HuggingFace SigLIP SO400M model
(google/siglip-so400m-patch14-224) purely as a text encoder — the vision side
comes entirely from the OmniVLA checkpoint.

Images are preprocessed identically to OmniVLA training:
  PrismaticImageProcessor, resize-naive, 224×224, mean/std=0.5/0.5, bicubic
  (index 1 of preprocessor_config.json in AsyncVLA_release).

Usage:
    python test_siglip.py path/to/image.jpg
    python test_siglip.py path/to/image.jpg --texts "navigate to the fence" "go to the gate"
    python test_siglip.py path/to/image.jpg --device mps --out heatmap.png

Requires:
    pip install grad-cam
    prismatic package on Python path (from the AsyncVLA repo)
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# ── Constants ──────────────────────────────────────────────────────────────────

ASYNCVLA_RELEASE_PATH = (
    "/Users/Lisa/Desktop/UoM/MC-MTRNENG/2026/Capstone/"
    "asyncvla-test/AsyncVLA/AsyncVLA_release"
)

# Standalone HuggingFace SigLIP SO400M — used only for its text encoder.
# Vision weights come from ASYNCVLA_RELEASE_PATH.
SIGLIP_HF_MODEL_ID = "google/siglip-so400m-patch14-224"

IMAGE_SIZE = 224
PATCH_SIZE = 14                              # SO400M uses 14-pixel patches
GRID_H = GRID_W = IMAGE_SIZE // PATCH_SIZE   # 16 × 16 = 256 patches

DEFAULT_TEXTS = [
    "navigate to the fence",
    "follow the fenceline",
    "go to the gate",
    "open grassland ahead",
]

# Matches preprocessor_config.json index 1 (SigLIP slot, resize-naive strategy).
SIGLIP_IMAGE_PROCESSOR = PrismaticImageProcessor(
    use_fused_vision_backbone=False,
    image_resize_strategy="resize-naive",
    input_sizes=[(3, IMAGE_SIZE, IMAGE_SIZE)],
    interpolations=["bicubic"],
    means=[(0.5, 0.5, 0.5)],
    stds=[(0.5, 0.5, 0.5)],
)


# ── Model loading ──────────────────────────────────────────────────────────────

def _register_hf_classes() -> None:
    """Register Prismatic custom classes with HuggingFace Auto APIs (idempotent)."""
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    except ValueError:
        pass  # already registered from a previous import


def load_siglip_featurizer(device: torch.device):
    """
    Load the full OmniVLA from AsyncVLA_release (same pattern as define_model()
    in run_asyncvla.py), extract vla.vision_backbone.siglip_featurizer, and free
    the rest of the VLA from memory.

    Returns the TIMM VisionTransformer in float32, eval mode, on `device`.
    """
    _register_hf_classes()
    print(f"Loading OmniVLA from {ASYNCVLA_RELEASE_PATH} ...")

    vla = AutoModelForVision2Seq.from_pretrained(
        ASYNCVLA_RELEASE_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Detach trained SigLIP featurizer; cast to float32 for stable GradCAM gradients
    siglip = vla.vision_backbone.siglip_featurizer.float().eval().to(device)

    n_params = sum(p.numel() for p in siglip.parameters()) / 1e6
    print(f"  SigLIP featurizer: {n_params:.0f} M params  (patch14, {GRID_H}×{GRID_W} grid)")

    del vla
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return siglip


# ── Model wrapper ──────────────────────────────────────────────────────────────

class SiglipGradCAMWrapper(torch.nn.Module):
    """Wraps the TIMM SigLIP featurizer to return language-image similarity logits.

    forward() calls forward_features() directly on the TIMM model, bypassing the
    monkey-patched .forward() that OmniVLA installs (which returns second-to-last
    layer features).  This ensures all blocks — including blocks[-1], the GradCAM
    target — are traversed in the forward pass.
    """

    def __init__(self, siglip_featurizer, text_features: torch.Tensor) -> None:
        super().__init__()
        self.siglip = siglip_featurizer
        self.register_buffer("text_features", text_features)  # (num_texts, d)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # forward_features() goes through all blocks; returns (B, num_patches, C)
        # SigLIP SO400M has no CLS/register tokens, so every token is a patch token
        x = self.siglip.forward_features(pixel_values)
        x = x.mean(dim=1)                               # global avg-pool → (B, C)
        x = x / x.norm(dim=-1, keepdim=True)
        return x @ self.text_features.T                 # (B, num_texts)


# ── GradCAM helpers ────────────────────────────────────────────────────────────

def reshape_transform(tensor, height: int = GRID_H, width: int = GRID_W):
    """Convert TIMM Block output (B, num_patches, C) → (B, C, H, W).

    TIMM Blocks return plain tensors (no tuple wrapping).  SigLIP SO400M has no
    class/register prefix tokens, so num_patches == height * width.  The extra
    check trims any unexpected prefix tokens defensively.
    """
    b, n, c = tensor.shape
    expected = height * width
    if n != expected:
        tensor = tensor[:, n - expected:, :]     # drop prefix tokens if present
    result = tensor.reshape(b, height, width, c)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# ── Core analysis ──────────────────────────────────────────────────────────────

def run_gradcam(
    image: Image.Image,
    texts: list,
    device: str = "cpu",
    out_path: str = "siglip_gradcam.png",
) -> None:
    """Run GradCAM for each text instruction and save a multi-panel figure."""

    dev = torch.device(device)

    # Vision: trained SigLIP featurizer from AsyncVLA_release
    siglip = load_siglip_featurizer(dev)

    # Text: standalone SigLIP SO400M (pretrained, used for text embeddings only)
    print(f"Loading text encoder from {SIGLIP_HF_MODEL_ID} ...")
    text_model    = AutoModel.from_pretrained(SIGLIP_HF_MODEL_ID).to(dev).eval()
    text_processor = AutoProcessor.from_pretrained(SIGLIP_HF_MODEL_ID)

    text_inputs = text_processor(
        text=texts, padding="max_length", return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        text_features = text_model.get_text_features(**text_inputs).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    del text_model   # free text model; keep only the embedding vectors

    wrapped = SiglipGradCAMWrapper(siglip, text_features).to(dev)

    # Last TIMM transformer block — output is a plain (B, 256, C) tensor
    target_layers = [siglip.blocks[-1]]

    # Apply the OmniVLA SigLIP training transform (resize-naive, mean/std=0.5)
    rgb          = image.convert("RGB")
    pixel_values = SIGLIP_IMAGE_PROCESSOR.apply_transform(rgb).unsqueeze(0).to(dev)

    # Raw image array for heatmap overlay (float32, [0, 1], (H, W, 3))
    img_arr = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

    ncols = len(texts) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))

    axes[0].imshow(img_arr)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    with GradCAM(
        model=wrapped,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    ) as cam:
        for i, text in enumerate(texts):
            targets       = [ClassifierOutputTarget(i)]
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
            grayscale_cam = grayscale_cam[0]                # (H, W) for first image

            vis = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
            ax  = axes[i + 1]
            ax.imshow(vis)
            ax.set_title(f'"{text}"', fontsize=8)
            ax.axis("off")

            with torch.no_grad():
                score = wrapped(pixel_values).squeeze()[i].item()
            ax.set_xlabel(f"sim = {score:.3f}", fontsize=7)

    fig.suptitle(
        "SigLIP Grad-CAM  ·  OmniVLA AsyncVLA_release weights",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SigLIP Grad-CAM: heatmap using OmniVLA AsyncVLA_release weights"
    )
    parser.add_argument("image", help="Path to input image (JPEG, PNG, …)")
    parser.add_argument(
        "--texts", nargs="+", default=DEFAULT_TEXTS,
        metavar="TEXT",
        help="Language instructions to visualise (one panel each)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device: cpu | cuda | mps  (default: cpu)",
    )
    parser.add_argument(
        "--out", default="siglip_gradcam.png",
        help="Output image path (default: siglip_gradcam.png)",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    run_gradcam(image, args.texts, args.device, args.out)


if __name__ == "__main__":
    main()
