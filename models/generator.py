import torch
import torch.nn as nn
from utils import to_signed
from config import *

class GatedConv2d(nn.Module):
    """Gated convolution layer for inpainting.
    Applies both a feature extraction conv and a gating mask conv.
    The gating mask controls which parts of the features are passed on,
    helping the network handle missing regions explicitly (better for arbitrary shaped masks).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(GatedConv2d, self).__init__()
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode='reflect'
        )
        self.gating_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode='reflect'
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.feature_conv(x)
        gate = self.sigmoid(self.gating_conv(x))
        return feature * gate

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, is_gated=IS_GATED):
    """Helper function based on IS_GATED flag to create either a gated or a standard convolution layer."""
    layers = []
    if is_gated:
        layers.append(GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        # No normalization here, since it is gated
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode='reflect'))
        layers.append(nn.InstanceNorm2d(out_channels, affine=True)) # --> replaced BatchNorm2d
        # Normalization added since no gates!
    return nn.Sequential(*layers)

class UpsamplePixelShuffle(nn.Module):
    """Upsamples by a factor of 2 using pixel shuffle instead of nearest neighbor."""
    def __init__(self, in_channels, out_channels):
        super(UpsamplePixelShuffle, self).__init__()
        self.conv = conv_block(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, is_gated=False)
        # Optional normalization (skip or try LayerNorm if needed)
        #self.norm = nn.InstanceNorm2d(out_channels * 4, affine=True)
        self.pre_shuffle = nn.LeakyReLU(0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        # A 1x1 convolution layer for channel refinement (sharpen details) (a tiny linear mixer for channels)
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.post_shuffle = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # removed normalization here to avoid overly blurred output
        # may try layer norm later if unstable!
        #x = self.norm(x) # apply normalization before pixel shuffle!
        x = self.pre_shuffle(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)
        x = self.post_shuffle(x)
        return x

# ---------------------
# Coarse Generator
# ---------------------
class CoarseGenerator(nn.Module):
    """Stage 1: Coarse inpainting generator (global structure).
    Generates a coarse prediction of missing layers."""
    def __init__(self, in_channels=4):
        super(CoarseGenerator, self).__init__()
        self.encoder = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels, G_HIDDEN, 5, stride=1, padding=2, padding_mode='reflect'), # 64x64
            nn.ReLU(inplace=True),
            # inplace ReLU is used to prevent 'Out of memory', do not use in case of an error
            # Source: https://discuss.pytorch.org/t/guidelines-for-when-and-why-one-should-set-inplace-true/50923
            # 2nd layer
            nn.Conv2d(G_HIDDEN, G_HIDDEN * 2, 3, stride=2, padding=1, padding_mode='reflect'), # 32x32
            nn.InstanceNorm2d(G_HIDDEN * 2, affine=True), # --> replaced BatchNorm2d
            nn.ReLU(inplace=True),
            # 3rd layer
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN * 4, 3, stride=2, padding=1, padding_mode='reflect'), # 16x16
            nn.InstanceNorm2d(G_HIDDEN * 4, affine=True), # --> replaced BatchNorm2d
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            # Dilated convolutions to expand receptive field without increasing resolution
            # 4th layer
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=2, dilation=2, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            # 5th layer
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=4, dilation=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            # 6th layer
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=8, dilation=8, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            # 7th layer
            UpsamplePixelShuffle(G_HIDDEN * 4, G_HIDDEN * 2),
            # 8th layer
            UpsamplePixelShuffle(G_HIDDEN * 2, G_HIDDEN),
            # 9th layer
            nn.Conv2d(G_HIDDEN,3, 3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
    def forward(self, x):
        # x expected at full resolution (e.g. 256x256)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# ---------------------
# Fine Generator
# ---------------------
class FineGenerator(nn.Module):
    """Stage 2: Fine inpainting generator (detail refinement).
    Refines the coarse output with gated convolutions.
    - Gating controls feature propagation
    - BatchNorm would mix statistics between masked/unmasked regions
    - Avoid BatchNorm for WGAN-GP as it can introduce correlation across batch samples and destabilize
    - Keeps mask-aware nature of layers
    Skips normalization in encoder&middle (Yu et al., 2019) to
    do not weaken the gating affect of gated con. But apply in decoder
    since representation is more filled and normalization helps stability."""
    def __init__(self, in_channels=4):
        super(FineGenerator, self).__init__()
        self.encoder = nn.Sequential(
            # 1st layer
            conv_block(in_channels, G_HIDDEN, 5, padding=2, is_gated=True), # 64x64
            nn.ReLU(inplace=True),
            # 2nd layer
            conv_block(G_HIDDEN, G_HIDDEN * 2, 3, stride=2, padding=1, is_gated=True), # 32x32
            nn.ReLU(inplace=True),
            # 3rd layer
            conv_block(G_HIDDEN * 2, G_HIDDEN * 4, 3, stride=2, padding=1, is_gated=True), # 16x16
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            # Dilated convolutions to expand receptive field without increasing resolution
            # 4th layer
            conv_block(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=2, dilation=2, is_gated=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 5th layer
            conv_block(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=4, dilation=4, is_gated=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 6th layer
            conv_block(G_HIDDEN * 4, G_HIDDEN * 4, 3, padding=8, dilation=8, is_gated=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder_128 = nn.Sequential( # 128x128 output
            # 7th layer
            UpsamplePixelShuffle(G_HIDDEN * 4, G_HIDDEN * 2)
        )
        self.to_rgb_128 = nn.Sequential(
            nn.Conv2d(G_HIDDEN * 2, 3, 3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        self.decoder_256 = nn.Sequential( # 256x256 output
            # 8th layer
            UpsamplePixelShuffle(G_HIDDEN * 2, G_HIDDEN),
            # 9th layer
            nn.Conv2d(G_HIDDEN, 3, 3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        #x = self.decoder(x)

        # Multi-scale supervision
        features_128 = self.decoder_128(x) # hidden at 128 x 128
        out_128 = self.to_rgb_128(features_128) # intermediate with RGB
        out_256 = self.decoder_256(features_128) # final fine output
        return out_256, out_128

# ---------------------
# Generator Wrapper
# ---------------------
class Generator(nn.Module):
    """Generator wrapper: Coarse (global) + Fine (detail)"""
    def __init__(self, in_channels=IMAGE_CHANNELS, mask_channels=1):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.total_channels = in_channels + mask_channels

        self.coarse_gen = CoarseGenerator(in_channels=self.total_channels)
        self.fine_gen = FineGenerator(in_channels=self.total_channels)

    def forward(self, image, mask):
        # Mask: keep background, replace masked area with mask channel
        # Masked input is created internally,
        # feed to generator only the original and the mask
        # scale the mask from [0,1] to [-1,1] to not bias the generator
        mask_scaled = to_signed(mask)
        known_regions = (1 - mask_scaled)
        masked_input = torch.cat([image * known_regions, mask_scaled], 1)

        # Stage 1: Coarse prediction
        coarse_output = self.coarse_gen(masked_input)

        # Blend coarse output with known regions, masked residual guidance on the coarse output
        blended_output = coarse_output * mask_scaled + image * known_regions

        # Stage 2: Fine prediction
        fine_input = torch.cat([blended_output, mask_scaled], 1)
        fine_output, out_128 = self.fine_gen(fine_input)

        # Add masked residual guidance on the fine output too!
        fine_output = fine_output * mask_scaled + image * known_regions

        return fine_output, out_128

# Create a helper function to load Generator
def load_generator(model_path):
    """Loads a pretrained Generator model.

    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator
    """