import os
import random
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import ImageOps
import PIL.Image as PILImage
from matplotlib import pyplot as plt
from torchvision import transforms

from config import LOCAL_PATCH_SIZE, SCALES, BATCH_SIZE, JITTER, SEED, EPS
from dataset import generate_square_mask


def downsample(img, scales=None):
    """Return a list of images downsampled to different scales."""
    if scales is None:
        scales = SCALES
    return [F.interpolate(img,
                          scale_factor=s,
                          mode='bilinear',
                          align_corners=False) for s in scales]

def dilation(x, size=3):
    # x = [B, 1, H, W] in [0, 1] i.e. mask_hole so mask 1, rest 0
    return F.max_pool2d(x, kernel_size=(2 * size + 1), stride=1, padding=size)

def erosion(x, size=3):
    return 1.0 - F.max_pool2d(1.0 - x, kernel_size=(2 * size + 1), stride=1, padding=size)

def get_ring(x, size=3, blur_kernel=0, normalize=False):
    dil = dilation(x, size)
    er = erosion(x, size)
    ring = {
        "inner": torch.clamp(x - er, 0.0, 1.0),
        "outer": torch.clamp(dil - x, 0.0, 1.0),
        "both": torch.clamp(dil - er, 0.0, 1.0)
    }

    if blur_kernel and blur_kernel > 1:
        padding = blur_kernel // 2
        for k in ring.keys():
            r = ring[k]
            r = F.avg_pool2d(r, kernel_size=blur_kernel, stride=1, padding=padding)
            if normalize:
                rmin = r.amin(dim=(2,3), keepdim=True)
                rmax = r.amax(dim=(2,3), keepdim=True)
                r = (r - rmin) / (rmax - rmin + EPS)
            ring[k] = r.clamp_(0,1)
    return ring

def crop_local_patch(images: torch.Tensor, masks_hole: torch.Tensor,
                     offsets: tuple[torch.Tensor, torch.Tensor],
                     pad_mode: str = 'reflection',
                     patch_size: int = LOCAL_PATCH_SIZE,
                     margin: int = 16) -> torch.Tensor:
    """
    Batched ROI crop that fully contains the mask with a margin,
    then applies offsets and resamples to [patch_size, patch_size].

    Args:
        images: tensor of shape [B, C, H, W],
        masks_hole: tensor of shape [B, 1, H, W], (1=hole, 0=known)
        offsets: (dy, dx) tensors of shape [B],
        pad_mode: 'reflection'
        patch_size: size of patches to be cropped
        margin: margin added around to form ROI
    Returns:
        patches: tensor of shape [B, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    device, dtype = images.device, images.dtype

    hole = (masks_hole > 0.5).squeeze(1)  # [B, H, W] bool
    has_hole = hole.any(dim=(1,2))  # [B] bool
    rows = hole.any(dim=2)  # [B, H] any across W
    cols = hole.any(dim=1)  # [B, W] any across H

    # Hole bbox (vectorized), all [B]
    top = rows.float().argmax(dim=1)
    bottom = (H - 1) - torch.flip(rows, [1]).float().argmax(dim=1)
    left = cols.float().argmax(dim=1)
    right = (W - 1) - torch.flip(cols, [1]).float().argmax(dim=1)
    top_f, bottom_f = top.to(dtype), bottom.to(dtype)
    left_f, right_f = left.to(dtype), right.to(dtype)

    # Center if no hole
    cy_no = torch.full_like(top, H // 2)
    cx_no = torch.full_like(left, W // 2)

    # Bbox center
    cy_bb = ((top + bottom).float()) * 0.5  # [B]
    cx_bb = ((left + right).float()) * 0.5  # [B]
    cy = torch.where(has_hole, cy_bb, cy_no.float())
    cx = torch.where(has_hole, cx_bb, cx_no.float())

    # ROI square one side = max(h,w) + 2*margin, at least patch_size
    h = (bottom - top + 1).clamp_min(1).float()
    w = (right - left + 1).clamp_min(1).float()
    side = torch.max(h, w) + 2.0 * float(margin)
    side = side.clamp_min(float(patch_size))

    # Scalar helpers for bounds
    H_t = torch.tensor(float(H), device=device, dtype=dtype)
    W_t = torch.tensor(float(W), device=device, dtype=dtype)
    margin_t = torch.tensor(float(margin), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    max_y0 = (H_t - side).clamp_min(zero) # [B]
    max_x0 = (W_t - side).clamp_min(zero) # [B]

    # Ensure ROI fully covers bbox after jitter
    # y0 must be in [top - margin, bottom + margin - side + 1] etc
    y0_min = (top_f - margin_t).clamp(min=zero, max=max_y0)
    y0_max = (bottom_f + margin_t - side + 1).clamp(min=zero, max=max_y0)
    x0_min = (left_f - margin_t).clamp(min=zero, max=max_x0)
    x0_max = (right_f + margin_t - side + 1).clamp(min=zero, max=max_x0)

    # Apply jitter and clamp to safe ranges
    dy, dx = offsets
    dy = dy.to(device=device, dtype=torch.float32)
    dx = dx.to(device=device, dtype=torch.float32)
    y0_c = (cy - 0.5 * (side - 1.0)).floor()
    x0_c = (cx - 0.5 * (side - 1.0)).floor()
    y0 = (y0_c + dy).clamp(y0_min, y0_max)
    x0 = (x0_c + dx).clamp(x0_min, x0_max)

    # Map ROI
    cy_roi = y0 + 0.5 * (side - 1.0)
    cx_roi = x0 + 0.5 * (side - 1.0)

    # Normalize by scale and translation
    sx = side / max(W - 1, 1)
    sy = side / max(H - 1, 1)
    tx = (2.0 * cx_roi / max(W - 1, 1)) - 1.0
    ty = (2.0 * cy_roi / max(H - 1, 1)) - 1.0

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=([B, C, patch_size, patch_size]), align_corners=False)
    patches = F.grid_sample(images, grid, mode="bilinear",
                            padding_mode=pad_mode, align_corners=False)
    return patches

def sample_offset(batch_size: int = BATCH_SIZE, jitter: int = JITTER, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size number of offsets in [-jitter, jitter]."""
    dy = torch.randint(-jitter, jitter + 1, (batch_size,), device=device)
    dx = torch.randint(-jitter, jitter + 1, (batch_size,), device=device)
    return dy, dx

def set_fixed(train_dataset, batch_size: int = BATCH_SIZE, device=None):
    """Create a fixed set of samples for consistent epoch visualization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    rand = random.Random(SEED)
    paths = getattr(train_dataset, 'crop_paths', [])[:batch_size]

    fixed_masked = []
    fixed_image = []
    fixed_mask_hole = []

    for p in paths:
        with PILImage.open(p) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            x = transform(img)
        _, height, width = x.shape

        mask = generate_square_mask(height, width, rand=rand, p_size=0.4)

        fixed_masked.append((x * mask).unsqueeze(0))  # create the masked sample
        fixed_image.append(x.unsqueeze(0))  # ground truth
        fixed_mask_hole.append((1.0 - mask).unsqueeze(0))  # hole mask (1=hole, 0=known)

    fixed_masked = torch.cat(fixed_masked, dim=0).to(device)
    fixed_image = torch.cat(fixed_image, dim=0).to(device)
    fixed_mask_hole = torch.cat(fixed_mask_hole, dim=0).to(device)
    return fixed_masked, fixed_image, fixed_mask_hole

def save_images(image, masked, composite, dim, nrow, out_dir, file_name):
    """Save given batch of images with masked and composite versions."""
    grid = torch.cat( # dimensions: (C,H,W)
        [image, masked, composite], dim)
    vutils.save_image(
        grid,
        os.path.join(out_dir, file_name),
        normalize=False,
        nrow=nrow
    )

def plot_loss(plot_type, plot_var, out_path):
    """Plot averaged loss curve per plot_type (Like DCGAN)"""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (per " + plot_type + ")")
    plt.plot(plot_var["totalG"], label="G (" + plot_type + ")")
    plt.plot(plot_var["totalD"], label="D (" + plot_type + ")")
    plt.plot(plot_var["globalD"], label="gD (" + plot_type + ")")
    plt.plot(plot_var["localD"], label="lD (" + plot_type + ")")
    plt.xlabel(plot_type.capitalize())
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_path, "loss_curve_" + plot_type + ".png"))
    plt.close()