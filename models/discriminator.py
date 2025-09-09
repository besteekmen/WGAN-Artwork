import torch
import torch.nn as nn
from config import *

# ---------------------
# Global Discriminator: judges full 256x256 image
# ---------------------
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer (Input: 3 x 256 x 256)
            nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False), # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            # Input layer does not have a batch normalization layer connected to it,
            # because it could lead to sample oscillation and model instability.

            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False), # 64x64
            #nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False), # 32x32
            #nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False), # 16x16
            #nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=False), # 13x13 -> 1x1 output
            # WGAN-GP: removed sigmoid as not using BCE, also normalizations are removed
            # nn.Sigmoid()
        )
    def forward(self, x):
        # Output shape: [batch_size, 1, 1, 1] -> flatten to [batch_size]
        #return self.main(x).view(-1) # same bu make sure the dimensions are controlled!
        return self.main(x).view(-1, 1).squeeze(1)

# ---------------------
# Local Discriminator: judges the inpainted mask patch (i.e. 128x128 or smaller)
# ---------------------
class LocalDiscriminator(nn.Module):
    # TODO: What about varying mask size? Does this help at all?
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer (Input: 3 x patch_size x patch_size, i.e. 3 x 128 x 128)
            nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False), # (patch_size/2)x(patch_size/2)
            nn.LeakyReLU(0.2, inplace=True),
            # Input layer does not have a batch normalization layer connected to it,
            # because it could lead to sample oscillation and model instability.

            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False), # (patch_size/4)x(patch_size/4)
            #nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False), # (patch_size/8)x(patch_size/8)
            #nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False), # (patch_size/16)x(patch_size/16)
            #nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=False), # 1x1 output
            # WGAN-GP: removed sigmoid as not using BCE, also BN layers are removed
            #nn.Sigmoid()
        )
    def forward(self, x):
        # return self.main(x).view(-1) # same bu make sure the dimensions are controlled!
        return self.main(x).view(-1, 1).squeeze(1)

# ---------------------
# Discriminator Wrapper: REMOVE!
# ---------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.global_discriminator = GlobalDiscriminator()
        self.local_discriminator = LocalDiscriminator()

    def forward(self, full_image, mask):
        """
        Args:
            full_image (torch.Tensor): shape [B, C, H, W] (inpainted or real)
            mask (torch.Tensor): shape [B, 1, H, W] (binary mask with 1 for masked area)
        Returns:
            global_out: shape [B]
            local_out: shape [B]
        """
        global_out = self.global_discriminator(full_image)
        local_patches = self.extract_local_patches(full_image, mask)
        local_out = self.local_discriminator(local_patches)
        return global_out, local_out

    @staticmethod
    def extract_local_patches(full_image, mask, patch_size=LOCAL_PATCH_SIZE):
        """Crop masked area for local discriminator"""
        B, C, H, W = full_image.size()
        patches = []

        for b in range(B):
            mask_b = mask[b, 0]
            ys, xs = mask_b.nonzero(as_tuple=True)

            if len(xs) == 0 or len(ys) == 0:
                # No mask, center crop
                center_x, center_y = W // 2, H // 2

            else:
                min_x, min_y = xs.min(), ys.min()
                max_x, max_y = xs.max(), ys.max()
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2

            left = max(0, center_x - patch_size // 2)
            top = max(0, center_y - patch_size // 2)
            right = min(left + patch_size, W)
            bottom = min(top + patch_size, H)
            #right = min(left + patch_size // 2, W)
            #bottom = min(top + patch_size // 2, H)

            if right - left < patch_size:
                left = max(right - patch_size, 0)
            if bottom - top < patch_size:
                top = max(bottom - patch_size, 0)

            patch = full_image[b:b+1, :, top:bottom, left:right]
            patches.append(patch)

        return torch.cat(patches, dim=0)