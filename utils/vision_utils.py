import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from config import LOCAL_PATCH_SIZE, SCALES, BATCH_SIZE, JITTER

def downsample(img, scales=None):
    """Return a list of images downsampled to different scales."""
    if scales is None:
        scales = SCALES
    return [F.interpolate(img,
                          scale_factor=s,
                          mode='bilinear',
                          align_corners=False) for s in scales]

def crop_local_patch(images: torch.Tensor, masks_hole: torch.Tensor,
                     offsets: tuple[torch.Tensor, torch.Tensor],
                     pad_mode: str = 'reflect',
                     patch_size: int = LOCAL_PATCH_SIZE) -> torch.Tensor:
    """
    images: tensor of shape [B, C, H, W],
    masks_hole: tensor of shape [B, 1, H, W],
    offsets: tuple[torch.Tensor, torch.Tensor], offset (dy, dx)
    returns: tensor of shape [B, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    half = patch_size // 2
    pad = half

    # Reflect-pad the whole batch
    padded = F.pad(images, (pad, pad, pad, pad), mode=pad_mode)
    hp, wp = H + 2 * pad, W + 2 * pad

    hole = (masks_hole.squeeze(1) > 0.5) # [B, H, W]
    rows = hole.any(dim=2).float() # [B, H] any across W
    cols = hole.any(dim=1).float() # [B, W] any across H

    # Hole bbox (vectorized), all [B]
    top = rows.argmax(dim=1)
    bottom = (H - 1) - torch.flip(rows, [1]).argmax(dim=1)
    left = cols.argmax(dim=1)
    right = (W - 1) - torch.flip(cols, [1]).argmax(dim=1)

    # Center (in the original image), then shift by pad
    cy = ((top + bottom) // 2) + pad # [B]
    cx = ((left + right) // 2) + pad # [B]

    dy, dx = offsets
    dy = dy.to(images.device)
    dx = dx.to(images.device)

    cy = torch.clamp(cy + dy, half, hp - half)
    cx = torch.clamp(cx + dx, half, wp - half)

    # Top-left corners in the padded image (exact center cropping, no clamping needed)
    y0 = cy - half # [B]
    x0 = cx - half # [B]

    # Convert small tensors to CPU ints once to avoid GPU sync points
    y0 = y0.to('cpu', non_blocking=True).tolist()
    x0 = x0.to('cpu', non_blocking=True).tolist()

    patches = []
    for b in range(B):
        yy, xx = y0[b], x0[b]
        patches.append(padded[b:b+1, :, yy:yy+patch_size, xx:xx+patch_size])

    return torch.cat(patches, dim=0)

def sample_offset(batch_size: int = BATCH_SIZE, jitter: int = JITTER, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size number of offsets in [-jitter, jitter]."""
    dy = torch.randint(-jitter, jitter + 1, (batch_size,), device=device)
    dx = torch.randint(-jitter, jitter + 1, (batch_size,), device=device)
    return dy, dx

def set_fixed(train_dataset, batch_size: int = BATCH_SIZE, device=None):
    """Create a fixed set of samples for consistent epoch visualization."""
    fixed_masked = []
    fixed_image = []
    fixed_mask_hole = []
    for i in range(min(batch_size, len(train_dataset))):
        img, mask = train_dataset[i]  # mask: [2, H, W] (1=known, 0=hole)
        # half or less irregular? same style pics?
        shape = torch.randint(0, 2, (1,), device=mask.device).item()
        mask = mask[shape].unsqueeze(0)  # mask: [1, H, W]

        fixed_masked.append((img * mask).unsqueeze(0))  # create the masked sample
        fixed_image.append(img.unsqueeze(0))  # ground truth
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