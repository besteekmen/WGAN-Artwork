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