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

def crop_roi(images: torch.Tensor, masks_hole: torch.Tensor,
             pad_mode: str = 'reflect',
             patch_size: int = LOCAL_PATCH_SIZE,
             base_margin: int = 16,
             jitter: int = 16,
             rand_extra: int = 16,
             coords: bool = False,
             to_cpu: bool = True
             ):
    """
    Returns a crop that fully contains the mask (with jitter),
    resized to patch_size, but the mask's relative position
    and scale vary per sample. If coords True, returns per-sample
    coords and pads, so same ROI can be cropped later.
    """
    B, C, H, W = images.shape
    out = []
    y0s, x0s, sides = [], [], []
    tops, bottoms, lefts, rights = [], [], [], [] # for padding

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

            # record coords (no outside pad here)
            side = (y1 - y0)
            out.append(crop)
            y0s.append(y0)
            x0s.append(x0)
            sides.append(side)
            tops.append(0)
            bottoms.append(0)
            lefts.append(0)
            rights.append(0)
            continue

        top, bottom = ys.min().item(), ys.max().item()
        left, right = xs.min().item(), xs.max().item()
        h, w = bottom - top + 1, right - left + 1

        # Randomize margin
        margin = base_margin + int(torch.randint(0, rand_extra+1, (1,)).item())
        side = max(h, w) + 2 * margin

        # Start with a square ROI centered on the hole bbox
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

        # record coords and pads
        y0s.append(y0)
        x0s.append(x0)
        sides.append(side)
        tops.append(pad_top)
        bottoms.append(pad_bottom)
        lefts.append(pad_left)
        rights.append(pad_right)

    patches = torch.cat(out, dim=0)

    if not coords:
        return patches

    device = images.device
    to_device = (lambda x: torch.tensor(x, device=device, dtype=torch.int64))
    coords = {
        "y0": to_device(y0s),
        "x0": to_device(x0s),
        "side": to_device(sides),
        "pad_top": to_device(tops),
        "pad_bottom": to_device(bottoms),
        "pad_left": to_device(lefts),
        "pad_right": to_device(rights)
    }
    if to_cpu:
        coords = {k: v.cpu() for k, v in coords.items()}
    return patches, coords

def crop_coords(images: torch.Tensor, coords: dict,
                patch_size: int = LOCAL_PATCH_SIZE, pad_mode: str = "reflect") -> torch.Tensor:
    """Returns the ROI crop from the given coordinates."""
    # Move coords to device
    device = images.device
    y0 = coords["y0"].to(device)
    x0 = coords["x0"].to(device)
    side = coords["side"].to(device)
    top = coords["pad_top"].to(device)
    bottom = coords["pad_bottom"].to(device)
    left = coords["pad_left"].to(device)
    right = coords["pad_right"].to(device)

    B = images.size(0)
    out = []

    for b in range(B):
        img = images[b:b+1]
        if (top[b] | bottom[b] | left[b] | right[b]).item() != 0:
            img = F.pad(img, (left[b].item(), right[b].item(), top[b].item(), bottom[b].item()), mode=pad_mode)

        y = y0[b].item()
        x = x0[b].item()
        s = side[b].item()
        roi = img[:, :, y:y+s, x:x+s]
        roi = F.interpolate(roi, (patch_size, patch_size), mode="bilinear", align_corners=False)
        out.append(roi)
    return torch.cat(out, dim=0)

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