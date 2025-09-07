import torch
import torch.nn as nn
from models.weights_init import weights_init_upsample
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

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    """Helper function based on IS_GATED flag to create either a gated or a standard convolution layer."""
    layers = []
    if IS_GATED:
        layers.append(GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        # No normalization here, since it is gated
    else:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        layers.append(nn.InstanceNorm2d(out_channels, affine=True)) # --> replaced BatchNorm2d
        # Normalization added since no gates!
    return nn.Sequential(*layers)

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
            nn.Conv2d(in_channels, G_HIDDEN, 5, stride=1, padding=2), # 64x64
            nn.ReLU(inplace=True),
            # inplace ReLU is used to prevent 'Out of memory', do not use in case of an error
            # Source: https://discuss.pytorch.org/t/guidelines-for-when-and-why-one-should-set-inplace-true/50923
            # 2nd layer
            nn.Conv2d(G_HIDDEN, G_HIDDEN * 2, 3, stride=2, padding=1), # 32x32
            nn.InstanceNorm2d(G_HIDDEN * 2, affine=True), # --> replaced BatchNorm2d
            nn.ReLU(inplace=True),
            # 3rd layer
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN * 4, 3, stride=2, padding=1), # 16x16
            nn.InstanceNorm2d(G_HIDDEN * 4, affine=True), # --> replaced BatchNorm2d
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
            nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.ReflectionPad2d(1), # added to avoid edge ringing
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(G_HIDDEN * 2, affine=True), # --> replaced BatchNorm2d
            nn.ReLU(inplace=True),
            # 8th layer
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN, 3, stride=1, padding=1),
            nn.InstanceNorm2d(G_HIDDEN, affine=True),  # --> replaced BatchNorm2d
            nn.ReLU(inplace=True),
            # 9th layer (to RGB)
            nn.Conv2d(G_HIDDEN,3, 3, padding=1),
            nn.Tanh()
        )

    def upsample_init(self):
        # Apply specific initialization to conv after upsample
        weights_init_upsample(self.decoder[1].weight, scale=2)
        weights_init_upsample(self.decoder[5].weight, scale=2)

    def forward(self, x):
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
        self.decoder = nn.Sequential( # Norm removed to preserve textures
            # 7th layer
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN * 2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 8th layer
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 9th layer (to RGB)
            nn.Conv2d(G_HIDDEN, 3, 3, padding=1),
            nn.Tanh()
        )

    def upsample_init(self):
        # Apply specific initialization to conv after upsample
        weights_init_upsample(self.decoder[1].weight, scale=2)
        weights_init_upsample(self.decoder[4].weight, scale=2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

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

    def upsample_init(self):
        # Call only after usual weight initialization!
        self.coarse_gen.upsample_init()
        self.fine_gen.upsample_init()

    def forward(self, image, mask):
        # mask: it is 1 in the missing area and 0 in the known area
        # feed to generator only the masked input
        #mask_scaled = to_signed(mask)
        masked_input = torch.cat([image * (1.0 - mask), mask], 1)

        # Stage 1: Coarse prediction
        coarse_output = self.coarse_gen(masked_input)

        # Blend coarse output with known regions
        blended_output = coarse_output * mask + image * (1.0 - mask)

        # Stage 2: Fine prediction
        fine_input = torch.cat([blended_output, mask], 1)
        fine_output = self.fine_gen(fine_input)

        return fine_output

# Create a helper function to load Generator
def load_generator(model_path):
    """Loads a pretrained Generator model."""
    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator