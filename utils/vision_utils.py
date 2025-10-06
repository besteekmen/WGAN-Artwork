import os
import random
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import ImageOps
import PIL.Image as PILImage
from matplotlib import pyplot as plt
from torchvision import transforms

from config import LOCAL_PATCH_SIZE, SCALES, BATCH_SIZE, SEED, EPS
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

def crop_roi(images: torch.Tensor, masks_hole: torch.Tensor,
             pad_mode: str = 'reflect',
             patch_size: int = LOCAL_PATCH_SIZE,
             base_margin: int = 16,
             jitter: int = 16,
             rand_extra: int = 16) -> torch.Tensor:
    """
    Returns a crop that fully contains the mask but the mask's
    relative position and scale vary per sample.
    returns: tensor of shape [B, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    out = []
    for b in range(B):
        m = masks_hole[b, 0] > 0.5
        ys, xs = torch.where(m)
        # if no hole, go to image center
        if ys.numel() == 0:
            y0 = max(0, H//2 - patch_size//2)
            y1 = min(H, y0 + patch_size)
            x0 = max(0, W//2 - patch_size//2)
            x1 = min(W, x0 + patch_size)
            crop = images[b:b+1, :, y0:y1, x0:x1]
            crop = F.interpolate(crop, (patch_size, patch_size), mode='bilinear', align_corners=False)
            out.append(crop)
            continue

        top, bottom = ys.min().item(), ys.max().item()
        left, right = xs.min().item(), xs.max().item()
        h, w = bottom - top + 1, right - left + 1

        # Randomize margin
        margin = base_margin + int(torch.randint(0, rand_extra+1, (1,)).item())
        side = max(h, w) + 2 * margin

        # Slack to move the ROI centered on the hole bbox
        slack_y = max(0, side - h)
        slack_x = max(0, side - w)

        # start with a square ROI centered on the hole bbox
        y0_c = (top + bottom - side) // 2
        x0_c = (left + right - side) // 2

        # Jitter but clamp to image bounds so the hole stay inside
        j_y = int(torch.randint(-jitter, jitter+1, (1,)).item())
        j_x = int(torch.randint(-jitter, jitter+1, (1,)).item())
        y0 = max(0, min(H - side, y0_c + j_y))
        x0 = max(0, min(W - side, x0_c + j_x))
        y1, x1 = y0 + side, x0 + side

        # If ROI goes out, reflect pad
        pad_top = max(0, -y0)
        pad_left = max(0, -x0)
        pad_bottom = max(0, y1 - H)
        pad_right = max(0, x1 - W)
        if any(p > 0 for p in (pad_left, pad_right, pad_top, pad_bottom)):
            img = F.pad(images[b:b+1], (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
            y0 += pad_top
            y1 += pad_top
            x0 += pad_left
            x1 += pad_left
        else:
            img = images[b:b+1]

        roi = img[:, :, y0:y1, x0:x1]
        roi = F.interpolate(roi, (patch_size, patch_size), mode='bilinear', align_corners=False)
        out.append(roi)
    return torch.cat(out, dim=0)

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