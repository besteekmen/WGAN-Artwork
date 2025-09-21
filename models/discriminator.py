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
            nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=True), # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            # Input layer does not have a batch normalization layer connected to it,
            # because it could lead to sample oscillation and model instability.

            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=True), # 64x64
            #nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=True), # 32x32
            #nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=True), # 16x16
            #nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=True), # 13x13
            # WGAN-GP: removed sigmoid as not using BCE, also normalizations are removed
            # nn.Sigmoid()
        )
    def forward(self, x):
        # view(-1) and view(-1, 1).squeeze(1) are the same, but make sure the dimensions are controlled!
        return self.main(x).mean(dim=(2,3)).view(-1, 1).squeeze(1) # [B, 1, 13, 13] -> [B] with averaged

# ---------------------
# Local Discriminator: judges the inpainted mask patch (i.e. 128x128 or smaller)
# ---------------------
class LocalDiscriminator(nn.Module):
    # TODO: What about varying mask size? Does this help at all?
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer (Input: 3 x patch_size x patch_size, i.e. 3 x 128 x 128)
            nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=True), # (patch_size/2)x(patch_size/2)
            nn.LeakyReLU(0.2, inplace=True),
            # Input layer does not have a batch normalization layer connected to it,
            # because it could lead to sample oscillation and model instability.

            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=True), # (patch_size/4)x(patch_size/4)
            #nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=True), # (patch_size/8)x(patch_size/8)
            #nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=True), # (patch_size/16)x(patch_size/16)
            #nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=True), # 5x5 (for patch_size=128)
            # WGAN-GP: removed sigmoid as not using BCE, also BN layers are removed
            #nn.Sigmoid()
        )
    def forward(self, x):
        # keep patch-level output (PatchGAN) pix2pix
        #return self.main(x).view(-1, 1).squeeze(1)
        out = self.main(x) # [B, 1, 5, 5]
        return out.view(out.size(0), -1) # [B, num_patches]