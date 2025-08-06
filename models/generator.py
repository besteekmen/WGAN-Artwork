import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# To toggle between gated and standard convolutions in the fine stage
IS_GATED = True

# Define gated convolution (to use if IS_GATED)
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(GatedConv2d, self).__init__()
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        self.gating_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.feature_conv(x)
        gate = self.sigmoid(self.gating_conv(x))
        return feature * gate

# Convolution helper function based on IS_GATED flag
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    if IS_GATED:
        return GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

# ---------------------
# Coarse Generator
# ---------------------
class CoarseGenerator(nn.Module):
    def __init__(self, in_channels=4):
        super(CoarseGenerator, self).__init__()
        self.encoder = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels, G_HIDDEN, 5, stride=1, padding=2), # 64x64
            nn.ReLU(inplace=True),
            # inplace ReLU is used to prevent 'Out of memory', do not use in case of an error
            # Source: https://discuss.pytorch.org/t/guidelines-for-when-and-why-one-should-set-inplace-true/50923
            # 2nd layer
            nn.Conv2d(G_HIDDEN, G_HIDDEN * 2, 3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(inplace=True),
            # 3rd layer
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN * 4, 3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            # Dilated convolutions to expand receptive field without increasing resolution
            # 4th layer
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # 5th layer
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            # 6th layer
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            # 7th layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(inplace=True),
            # 8th layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, stride=2, padding=1),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(inplace=True),
            # 9th layer
            nn.Conv2d(G_HIDDEN,3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# ---------------------
# Fine Generator
# ---------------------
class FineGenerator(nn.Module):
    def __init__(self, in_channels=4):
        super(FineGenerator, self).__init__()
        self.encoder = nn.Sequential(
            # 1st layer
            conv_block(in_channels, G_HIDDEN, 5, padding=2), # 64x64
            nn.ReLU(inplace=True),
            # 2nd layer
            conv_block(G_HIDDEN, G_HIDDEN * 2, 3, stride=2, padding=1), # 32x32
            nn.ReLU(inplace=True),
            # 3rd layer
            conv_block(G_HIDDEN * 2, G_HIDDEN * 4, 3, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            # Dilated convolutions to expand receptive field without increasing resolution
            # 4th layer
            conv_block(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # 5th layer
            conv_block(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            # 6th layer
            conv_block(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            # 7th layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(inplace=True),
            # 8th layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, stride=2, padding=1),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(inplace=True),
            # 9th layer
            nn.Conv2d(G_HIDDEN,3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Full generator with coarse and fine stages
class Generator(nn.Module):
    def __init__(self, in_channels=IMAGE_CHANNELS, mask_channels=1):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.total_channels = in_channels + mask_channels

        self.coarse_gen = CoarseGenerator(in_channels=self.total_channels)
        self.fine_gen = FineGenerator(in_channels=self.total_channels)

    def forward(self, image, mask):
        # Mask: keep background, replace masked area with mask channel
        masked_input = torch.cat([image * (1 - mask), mask], 1)

        # Stage 1: Coarse prediction
        coarse_output = self.coarse_gen(masked_input)

        # Blend coarse output with known regions
        blended_output = coarse_output * mask + image * (1 - mask)

        # Stage 2: Fine prediction
        fine_input = torch.cat([blended_output, mask], 1)
        fine_output = self.fine_gen(fine_input)

        return fine_output

# Create a helper function to load Generator
def load_generator(model_path):
    """Loads a pretrained Generator model.

    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator
    """