import torch
import torch.nn.functional as F
from config import LOCAL_PATCH_SIZE, SCALES

def downsample(img, scales=None):
    """Return a list of images downsampled to different scales."""
    if scales is None:
        scales = SCALES
    return [F.interpolate(img,
                          scale_factor=s,
                          mode='bilinear',
                          align_corners=False) for s in scales]

def crop_local_patch(images: torch.Tensor, masks_hole: torch.Tensor, patch_size: int = LOCAL_PATCH_SIZE) -> torch.Tensor:
    """
    images: tensor of shape [B, C, H, W],
    masks_hole: tensor of shape [B, 1, H, W],
    returns: tensor of shape [B, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    pad = patch_size // 2

    # Reflect-pad the whole batch
    padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')

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

    # Top-left corners in the padded image (exact center cropping, no clamping needed)
    y0 = cy - patch_size // 2 # [B]
    x0 = cx - patch_size // 2 # [B]

    # Convert small tensors to CPU ints once to avoid GPU sync points
    y0 = y0.to('cpu', non_blocking=True).tolist()
    x0 = x0.to('cpu', non_blocking=True).tolist()

    patches = []
    for b in range(B):
        yy, xx = y0[b], x0[b]
        patches.append(padded[b:b+1, :, yy:yy+patch_size, xx:xx+patch_size])

    return torch.cat(patches, dim=0)