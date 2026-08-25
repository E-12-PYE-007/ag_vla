"""
siglip_maps.py

SigLIP SO400M saliency maps for the AsyncVLA / OmniVLA vision encoder.

Compares ℱ₆ Grad-CAM on OmniVLA's fine-tuned SigLIP tower vs stock SigLIP, so the
effect of the OmniVLA fine-tune on text-conditioned saliency is visible directly.

For each (instruction, image) pair it renders a 3-panel figure:
    Original | Grad-CAM · OmniVLA (fine-tuned) | Grad-CAM · stock SigLIP
both attribution panels are ℱ₆ Grad-CAM (gradient-weighted last-block patch
features, CLIP-ES lineage) targeting SigLIP's text-image cosine similarity.

(The ℱ₂ Chefer attention-rollout code is kept but disabled below — on SigLIP it is
dominated by attention sinks and did not reveal usable structure.)

Outputs → results_async_vs_stock/.  Run in the asyncvla conda env (prismatic,
torch 2.2.0, captum):
    python siglip_maps.py
"""

from pathlib import Path

import numpy as np
import timm
import torch
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from PIL import Image
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModel, AutoModelForVision2Seq, AutoProcessor,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# ── Constants ────────────────────────────────────────────────────────────────
ASYNCVLA_RELEASE_PATH = "/home/vla-cap/AsyncVLA/AsyncVLA_release"
SIGLIP_HF_MODEL_ID    = "google/siglip-so400m-patch14-224"
STOCK_TIMM_ID         = "vit_so400m_patch14_siglip_224"

IMAGE_SIZE = 224
PATCH_SIZE = 14
GRID_H = GRID_W = IMAGE_SIZE // PATCH_SIZE   # 16 × 16 = 256 patches
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION_DIRS = {
    "Fence":   "/home/vla-cap/vla-capstone-ros-ws/images/fence_right_flat",
    "Follow the fence on your left":    "/home/vla-cap/vla-capstone-ros-ws/images/fence_left_flat",
    "Road":            "/home/vla-cap/vla-capstone-ros-ws/images/road",
    "Shed": "/home/vla-cap/vla-capstone-ros-ws/images/shed",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESULTS_ROOT = Path(__file__).resolve().parent

# SigLIP preprocessing: resize-naive 224×224, bicubic, mean/std = 0.5
SIGLIP_IMAGE_PROCESSOR = PrismaticImageProcessor(
    use_fused_vision_backbone=False,
    image_resize_strategy="resize-naive",
    input_sizes=[(3, IMAGE_SIZE, IMAGE_SIZE)],
    interpolations=["bicubic"],
    means=[(0.5, 0.5, 0.5)],
    stds=[(0.5, 0.5, 0.5)],
)


# ── Model / text loading ─────────────────────────────────────────────────────
def _register_hf_classes() -> None:
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    except ValueError:
        pass


def load_omnivla_siglip(device: torch.device):
    """OmniVLA's fine-tuned SigLIP SO400M tower (fused_featurizer) from AsyncVLA_release."""
    _register_hf_classes()
    print(f"Loading OmniVLA from {ASYNCVLA_RELEASE_PATH} ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        ASYNCVLA_RELEASE_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    siglip = vla.vision_backbone.fused_featurizer.float().eval().to(device)
    print(f"  SigLIP SO400M: {sum(p.numel() for p in siglip.parameters())/1e6:.0f} M params")
    del vla
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return siglip


def load_stock_siglip(device: torch.device):
    """Stock pretrained timm SigLIP SO400M."""
    print(f"Loading stock timm SigLIP ({STOCK_TIMM_ID}) ...")
    m = timm.create_model(STOCK_TIMM_ID, pretrained=True, num_classes=0).float().eval().to(device)
    print(f"  SigLIP SO400M: {sum(p.numel() for p in m.parameters())/1e6:.0f} M params")
    return m


_TEXT_ENCODER = {}


def get_text_embedding(text: str, device: torch.device) -> torch.Tensor:
    """Normalised (1, D) text embedding from the HF SigLIP text tower (encoder cached)."""
    if "model" not in _TEXT_ENCODER:
        print(f"Loading text encoder ({SIGLIP_HF_MODEL_ID}) ...")
        _TEXT_ENCODER["model"] = AutoModel.from_pretrained(SIGLIP_HF_MODEL_ID).to(device).eval()
        _TEXT_ENCODER["proc"]  = AutoProcessor.from_pretrained(SIGLIP_HF_MODEL_ID)
    inputs = _TEXT_ENCODER["proc"](text=[text], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        emb = _TEXT_ENCODER["model"].get_text_features(**inputs).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb   # (1, D)


# ── Similarity wrapper ───────────────────────────────────────────────────────
class SiglipSimWrapper(torch.nn.Module):
    """
    similarity = cos( forward_head(forward_features(x)) , text_emb ).

    Pools through SigLIP's learned attention-pool head so the image embedding lands
    in the joint contrastive space, then takes cosine with the text embedding.
    """

    def __init__(self, tower, text_emb: torch.Tensor) -> None:
        super().__init__()
        self.siglip = tower
        self.register_buffer("text_emb", text_emb)   # (1, D)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.siglip.forward_head(self.siglip.forward_features(pixel_values))
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return (pooled @ self.text_emb.T).squeeze(-1)


# ── ℱ₆ Grad-CAM ──────────────────────────────────────────────────────────────
def compute_gradcam_map(wrapper, pixel_values: torch.Tensor) -> np.ndarray:
    """
    Grad-CAM on the last ViT block's patch tokens (ℱ₆), targeting the text-image
    similarity.  One forward + one backward:

        alpha_c = mean_n ∂sim/∂A[n, c];  cam[n] = ReLU( Σ_c alpha_c · A[n, c] )
    """
    store = {}
    layer = wrapper.siglip.blocks[-1]

    def fwd_hook(_m, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        wrapper.zero_grad(set_to_none=True)
        wrapper(pixel_values).sum().backward()
    finally:
        handle.remove()

    A = store["act"][0].detach()   # (N, C)
    G = store["grad"][0].detach()  # (N, C)
    alpha = G.mean(dim=0)
    cam = torch.relu((alpha * A).sum(dim=-1)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.reshape(GRID_H, GRID_W)


# ── ℱ₂ grad-weighted attention rollout (Chefer) — DISABLED ────────────────────
# Commented out: on SigLIP the rollout is dominated by attention sinks and did not
# reveal usable structure. Kept for reference; the workflow now compares Grad-CAM
# on OmniVLA vs stock SigLIP instead.
r'''
def _capture_block_attn(m, x, sink):
    """timm ViT block-attention forward that exposes the softmax matrix (with grad)."""
    B, N, C = x.shape
    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = m.q_norm(q), m.k_norm(k)
    attn = ((q * m.scale) @ k.transpose(-2, -1)).softmax(dim=-1)   # (B, heads, N, N)
    attn.retain_grad()
    sink.append(attn)
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    return m.proj_drop(m.proj(out))


def _capture_pool_attn(ap, x, holder):
    """AttentionPoolLatent forward that exposes the query→patch attention (with grad)."""
    B, N, C = x.shape
    if getattr(ap, "pos_embed", None) is not None:
        x = x + ap.pos_embed.unsqueeze(0).to(dtype=x.dtype)
    q = ap.q(ap.latent.expand(B, -1, -1)) \
        .reshape(B, ap.latent_len, ap.num_heads, ap.head_dim).transpose(1, 2)
    kv = ap.kv(x).reshape(B, N, 2, ap.num_heads, ap.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    q, k = ap.q_norm(q), ap.k_norm(k)
    attn = ((q * ap.scale) @ k.transpose(-2, -1)).softmax(dim=-1)   # (B, heads, latent_len, N)
    attn.retain_grad()
    holder["pool"] = attn
    out = (attn @ v).transpose(1, 2).reshape(B, ap.latent_len, C)
    out = ap.proj_drop(ap.proj(out))
    out = out + ap.mlp(ap.norm(out))
    return out.mean(1) if getattr(ap, "pool", "token") == "avg" else out[:, 0]


def compute_chefer_rollout_map(wrapper, pixel_values: torch.Tensor) -> np.ndarray:
    """
    Grad-weighted attention rollout across ALL ViT blocks (Chefer et al., CVPR 2021),
    read out through the attention-pool head.

    Idea: a single attention layer is a noisy explanation, so combine the (ℱ₂)
    attention matrices from every layer into one token-to-token relevance matrix R,
    weighting each attention edge by how much it moves the target (our text-image
    cosine).  The final map is the input patches most responsible for that match,
    accounting for how information routes through the whole stack.

    Per layer l  —  A_l is the softmax attention (heads, N, N); A_l[i,j] = how much
    query token i attends to key token j:

        C_l = ReLU( A_l ⊙ ∂sim/∂A_l ),  mean over heads              → (N, N)

        · ⊙ (Hadamard) keeps an edge only if it is active AND raises the similarity.
        · ReLU drops edges that push the target down.
        · C_l[i,j] = contribution of edge "i attends to j" to the match, at layer l.

    Rollout across layers  —  fold the C_l into one relevance matrix R:

        Â_l = rownorm( C_l + I );   R = Â_l @ R    (R starts at I)

        · The "+ I" is the residual/skip connection: a token also carries its own
          representation forward, not only what attention routes to it.
        · Multiplying Â_L · … · Â_1 traces multi-hop flow input → … → output.
        · R[i,j] ≈ how much INPUT patch j ends up influencing OUTPUT token i,
          gradient-weighted toward the instruction.

    Read-out  —  SigLIP has no [CLS] token, so instead of taking a CLS row of R we
    read out through the attention-pool head (the query that forms the image
    embedding used in the similarity):

        p = ReLU( A_pool ⊙ ∂sim/∂A_pool ), mean over heads           → (1, N)
        saliency_j = Σ_i p_i · R[i, j]      (= p @ R)                → per patch

    Then reshape N=256 → 16×16 and min-max normalize.

    Contrast with compute_gradcam_map (ℱ₆): that attributes to token *features* at
    one block; this attributes to attention *edges* across all blocks — "where the
    information flow concentrates" vs "which patch's content drives the score."

    Implementation note: SigLIP uses fused SDPA, which never materializes the
    softmax matrix, so every block/pool attention is temporarily patched to the
    explicit-softmax path (weights kept in the graph with retain_grad) and restored
    afterwards in the finally-block.
    """
    blocks = wrapper.siglip.blocks
    ap = wrapper.pool_head
    dev = pixel_values.device

    sink, holder = [], {}
    for b in blocks:
        b.attn.forward = (lambda x, m=b.attn: _capture_block_attn(m, x, sink))
    ap.forward = (lambda x, a=ap: _capture_pool_attn(a, x, holder))
    try:
        wrapper.zero_grad(set_to_none=True)
        wrapper(pixel_values).sum().backward()

        N = sink[0].shape[-1]
        eye = torch.eye(N, device=dev)
        R = eye.clone()
        for A in sink:
            C = torch.relu(A * A.grad).mean(dim=1)[0]      # (N, N) mean over heads
            Ahat = C + eye
            Ahat = Ahat / Ahat.sum(dim=-1, keepdim=True)
            R = Ahat @ R
        Ap = holder["pool"]                                # (B, heads, 1, N)
        p = torch.relu(Ap * Ap.grad).mean(dim=1)[0, 0]     # (N,)
        p = p / (p.sum() + 1e-8)
        sal = (p @ R).detach().cpu().numpy()
    finally:
        for b in blocks:
            del b.attn.forward
        del ap.forward

    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.reshape(GRID_H, GRID_W)
'''  # end DISABLED Chefer ℱ₂ rollout


# ── Plotting ─────────────────────────────────────────────────────────────────
def upsample_map(heatmap: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(target_h, target_w),
                                        mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def plot_gradcam_compare(img_arr, gradcam_async, gradcam_stock, text, out_path):
    """Three panels: original | Grad-CAM OmniVLA (fine-tuned) | Grad-CAM stock SigLIP."""
    h, w = img_arr.shape[:2]
    a_attr = upsample_map(gradcam_async, h, w)[..., np.newaxis]
    s_attr = upsample_map(gradcam_stock, h, w)[..., np.newaxis]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    viz.visualize_image_attr(
        a_attr, img_arr, method="blended_heat_map", sign="positive",
        cmap="plasma", alpha_overlay=0.55, show_colorbar=True,
        title="OmniVLA SigLIP",
        plt_fig_axis=(fig, axes[1]), use_pyplot=False,
    )
    axes[1].title.set_fontsize(11)
    axes[1].title.set_fontweight("bold")

    viz.visualize_image_attr(
        s_attr, img_arr, method="blended_heat_map", sign="positive",
        cmap="plasma", alpha_overlay=0.55, show_colorbar=True,
        title="Original SigLIP",
        plt_fig_axis=(fig, axes[2]), use_pyplot=False,
    )
    axes[2].title.set_fontsize(11)
    axes[2].title.set_fontweight("bold")

    fig.suptitle(f'{text}\nGrad-CAM on Feature Representation on OmniVLA and Original SigLIP',
                 fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Runner ───────────────────────────────────────────────────────────────────
def _images_for(dir_path: str):
    return sorted(p for p in Path(dir_path).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_compare(dev):
    """Grad-CAM on OmniVLA vs stock SigLIP, side by side, per instruction/image."""
    async_tower = load_omnivla_siglip(dev)
    stock_tower = load_stock_siglip(dev)
    out_root = RESULTS_ROOT / "results_aync_stock_gradcam_only"

    for text, img_dir in INSTRUCTION_DIRS.items():
        images = _images_for(img_dir)
        if not images:
            print(f"[skip] no images in {img_dir}")
            continue
        print(f'\n── "{text}" ({len(images)} images) ──')
        te = get_text_embedding(text, dev)
        w_async = SiglipSimWrapper(async_tower, te).to(dev)
        w_stock = SiglipSimWrapper(stock_tower, te).to(dev)
        out_dir = out_root / text.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            print(f"  {img_path.name}", end="  ")
            rgb = Image.open(img_path).convert("RGB")
            px = SIGLIP_IMAGE_PROCESSOR.apply_transform(rgb).unsqueeze(0).to(dev)
            img_arr = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype("float32") / 255.0
            with torch.no_grad():
                print(f"sim async={w_async(px).item():+.4f} stock={w_stock(px).item():+.4f}", end="  ")
            gc_async = compute_gradcam_map(w_async, px)
            gc_stock = compute_gradcam_map(w_stock, px)
            out_path = out_dir / f"{img_path.stem}_siglip_maps.png"
            plot_gradcam_compare(img_arr, gc_async, gc_stock, text, str(out_path))
            print(f"→ {out_path.name}")


if __name__ == "__main__":
    run_compare(torch.device(DEVICE))
