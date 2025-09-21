import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init_normal(m):
    """Initialize trainable parameters:
    as in DCGAN paper, extended for InstanceNorm
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_upsample(weight: torch.Tensor, scale: int=2):
    """Initialize convolution weights after 'Upsample':
    It mimics clean upsampling and avoid checkerboard at start."""
    if weight.dim() != 4:
        return # skip if not Conv2d weight

    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    # For out_channels = 64, scale=2 --> sub = 16 so 4 groups for upsampling by 2
    sub_channels = out_channels // (scale ** 2)

    if out_channels % (scale ** 2) != 0:
        raise ValueError(f"For upsample init, out_channels must be divisible by {scale**2}, got {out_channels}.")

    weight.data.zero_()
    for i in range(scale ** 2):
        init.kaiming_normal_(
            weight.data[i * sub_channels:(i + 1) * sub_channels, :, :, :],
            nonlinearity='relu'
        )

def bias_init_gate(m):
    """Initialize gated convolution bias"""
    if hasattr(m, 'gating_conv') and isinstance(m.gating_conv, nn.Conv2d):
        if m.gating_conv.bias is not None:
            #with torch.no_grad():
            init.constant_(m.gating_conv.bias, 1.0)
