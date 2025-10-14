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
            # No normalization here anyways, to avoid instability!

            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False), # 64x64
            nn.LeakyReLU(0.2, inplace=True),

            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False), # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False), # 16x16
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=False), # 13x13
            # WGAN-GP: removed sigmoid as not using BCE, also normalizations are removed (was BN)
            # nn.Sigmoid()
        )
    def forward(self, x):
        """Global discriminator.
        Gets the full size 256x256 image (original and fake both),
        and returns a single averaged critic score for each. Unlike Vanilla GAN,
        this is not True, False but a critic score that could also be negative.
        """
        # view(-1) and view(-1, 1).squeeze(1) are the same, but make sure the dimensions are controlled!
        return self.main(x).mean(dim=(2,3)).view(-1, 1).squeeze(1) # [B, 1, 13, 13] -> [B] with averaged

# --------------------
# Local Discriminator: judges the inpainted mask patch (i.e. 128x128 or smaller)
# --------------------
class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.b1 = nn.Sequential(
            # 1st layer (Input: 3 x patch_size x patch_size, i.e. 3 x 128 x 128)
            nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False),
            # (patch_size/2)x(patch_size/2)
            nn.LeakyReLU(0.2, inplace=True),
            # Input layer does not have a batch normalization layer connected to it,
            # because it could lead to sample oscillation and model instability.
        )
        self.b2 = nn.Sequential(
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # (patch_size/4)x(patch_size/4)
            # nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b3 = nn.Sequential(
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False),
            # (patch_size/8)x(patch_size/8)
            # nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b4 = nn.Sequential(
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False),
            # (patch_size/16)x(patch_size/16)
            # nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Sequential(
            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),  # 5x5 (for patch_size=128)
            # WGAN-GP: removed sigmoid as not using BCE, also BN layers are removed
            # nn.Sigmoid()
        )
    def forward(self, x, return_features: bool = False):
        """Local discriminator.
        Gets a cropped patch from the image (both original and fake),
        and returns either only a score set (for patches),
        or also intermediate feature maps for feature matching.
        """
        f1 = self.b1(x)         # [B, D, H/2, W/2]
        f2 = self.b2(f1)         # [B, 2D, H/4, W/4]
        f3 = self.b3(f2)         # [B, 4D, H/8, W/8]
        f4 = self.b4(f3)         # [B, 8D, H/16, W/16]
        logits = self.out(f4)   # [B, 1, 5, 5] for 128x128
        score = logits.view(logits.size(0), -1) # [B, num_patches]

        if not return_features:
            return score

        feats = [torch.mean(f3, dim=(2,3)), torch.mean(f4, dim=(2,3))] # each [B, C]
        return score, feats