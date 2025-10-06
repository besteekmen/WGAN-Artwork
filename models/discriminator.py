import torch
import torch.nn as nn
from config import *

# ---------------------
# Global Discriminator: judges full 256x256 image
# ---------------------
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        # 1st layer (Input: 3 x 256 x 256)
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False) # 128x128
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 2nd layer
        self.conv2 = nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False) # 64x64
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # 3rd layer
        self.conv3 = nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False)  # 32x32
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # 4th layer
        self.conv4 = nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False)  # 16x16
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        # Output layer
        self.out = nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=False) # 13x13
        # WGAN-GP: removed sigmoid as not using BCE, also normalizations are removed
        # nn.Sigmoid()

    def forward(self, x, features=False):
        f1 = self.act1(self.conv1(x))
        f2 = self.act2(self.conv2(f1))
        f3 = self.act3(self.conv3(f2))
        f4 = self.act4(self.conv4(f3))
        out = self.out(f4)
        score = out.mean(dim=(2,3)).view(-1, 1).squeeze(1) # [B, 1, 13, 13] -> [B] with averaged

        if features:
            return score, [f2, f4]
        return score
# ---------------------
# Local Discriminator: judges the inpainted mask patch (i.e. 128x128 or smaller)
# ---------------------
class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
            # 1st layer (Input: 3 x patch_size x patch_size, i.e. 3 x 128 x 128)
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, D_HIDDEN, kernel_size=4, stride=2, padding=1, bias=False) # (patch_size/2)x(patch_size/2)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 2nd layer
        self.conv2 = nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, kernel_size=4, stride=2, padding=1, bias=False) # (patch_size/4)x(patch_size/4)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # 3rd layer
        self.conv3 = nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, kernel_size=4, stride=2, padding=1, bias=False) # (patch_size/8)x(patch_size/8)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # 4th layer
        self.conv4 = nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, kernel_size=4, stride=2, padding=1, bias=False) # (patch_size/16)x(patch_size/16)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        # Output layer
        self.out = nn.Conv2d(D_HIDDEN * 8, 1, kernel_size=4, stride=1, padding=0, bias=False) # 5x5 (for patch_size=128)
        # WGAN-GP: removed sigmoid as not using BCE, also BN layers are removed
        # nn.Sigmoid()

    def forward(self, x, features=False):
        # keep patch-level output (PatchGAN) pix2pix
        #return self.main(x).view(-1, 1).squeeze(1)
        #out = self.main(x)
        f1 = self.act1(self.conv1(x))
        f2 = self.act2(self.conv2(f1))
        f3 = self.act3(self.conv3(f2))
        f4 = self.act4(self.conv4(f3))
        out = self.out(f4) # [B, 1, 5, 5] for 128x128 patch
        score = out.view(out.size(0), -1) # [B, num_patches]

        if features:
            return score, [f2, f4]
        return score