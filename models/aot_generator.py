import torch
import torch.nn as nn
import torch.nn.functional as F
from models.weights_init import weights_init_normal
from config import *

class AOTGenerator(nn.Module):
    def __init__(self, in_channels=4):
        super(AOTGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            # 1st layer
            nn.Conv2d(in_channels, G_HIDDEN, 7),
            nn.ReLU(inplace=True),
            # inplace ReLU is used to prevent 'Out of memory', do not use in case of an error
            # Source: https://discuss.pytorch.org/t/guidelines-for-when-and-why-one-should-set-inplace-true/50923
            # 2nd layer
            nn.Conv2d(G_HIDDEN, G_HIDDEN * 2, 4, stride=2, padding=1),  # [B, 128, 128, 128]
            nn.ReLU(inplace=True),
            # 3rd layer
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN * 4, 4, stride=2, padding=1),  # [B, 256, 64, 64]
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            AOTBlock(G_HIDDEN * 4),
            AOTBlock(G_HIDDEN * 4),
            AOTBlock(G_HIDDEN * 4),
            AOTBlock(G_HIDDEN * 4)
        )
        self.decoder = nn.Sequential(
            # 7th layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, stride=2, padding=1, bias=True),
            #nn.Conv2d(G_HIDDEN * 2, G_HIDDEN * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            # 8th layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, stride=2, padding=1, bias=True),
            #nn.Conv2d(G_HIDDEN, G_HIDDEN, 3, padding=1),
            nn.ReLU(inplace=True),
            # 9th layer (to RGB)
            nn.Conv2d(G_HIDDEN, 3, 3, padding=1)
        )
        self.apply(weights_init_normal)

        for m in self.modules():
            if isinstance(m, ARTBlock):
                m.reset()

    def forward(self, x, mask):
        """Gets an image and a mask to forward.
        Arguments:
            image: [B, 3, H, W] values in [-1, 1]
            mask: [B, 1, H, W] values in [0, 1], (1=hole, 0=known)
        """
        masked_input = x * (1.0 - mask)
        x = torch.cat((masked_input, mask), dim=1)
        x = self.encoder(x)
        x = self.middle(x)

        # Context aware AdaIN
        #hole = F.max_pool2d(mask, kernel_size=4, stride=4, ceil_mode=True) # [B, 1, H, W]
        #known = 1.0 - hole # known pixels (0=hole, 1=known)

        #def masked_moments(feat, m, eps=1e-6):
        #    denom = m.sum((2,3), keepdim=True).clamp_min(1.0)
        #    mu = (feat * m).sum((2,3), keepdim=True) / denom
        #    var = ((feat - mu) ** 2 * m).sum((2,3), keepdim=True) / denom
        #    std = (var + eps).sqrt()
        #    return mu, std

        #mu_k, std_k = masked_moments(x, known) # stats of surround
        #mu_h, std_h = masked_moments(x, hole) # stats of hole

        #h_hole = (x - mu_h) / std_h * std_c + mu_c
        #x = x * (1.0 - hole) + h_hole * hole

        x = self.decoder(x)
        return torch.tanh(x)

def aot_layer_norm(features):
    mean = features.mean((2, 3), keepdim=True)
    std = features.std((2, 3), keepdim=True) + 1e-9
    # return 3.0 * (features - mean) / std # lower gain
    features = 2 * (features - mean) / std - 1
    features = 5 * features
    return features

class AOTBlock(nn.Module):
    def __init__(self, dim):
        super(AOTBlock, self).__init__()
        self.block0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=4),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.ReflectionPad2d(8),
            nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=8),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

        # learnable branch weights
        self.alpha = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, x):
        out0 = self.block0(x)
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)

        scale = 1.0 + 0.25 * torch.tanh(self.alpha) # range [0.75, 1.25]
        out0 = out0 * scale[:,0:1]
        out1 = out1 * scale[:,1:2]
        out2 = out2 * scale[:,2:3]
        out3 = out3 * scale[:,3:4]

        out = torch.cat([out0, out1, out2, out3], dim=1)
        out = self.fuse(out)
        mask = torch.sigmoid(aot_layer_norm(self.gate(x)))
        return x * (1 - mask) + out * mask


def AOTfilter(channel, kernel, norm=None):
    """Returns a frozen depthwise 3x3 convolution with the given kernel size."""
    k = torch.tensor(kernel, dtype=torch.float32)
    if norm is not None:
        k = k / float(norm)
    conv = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
    with torch.no_grad():
        weight = k.view(1, 1, 3, 3).expand(channel, 1, 3, 3).clone()
        conv.weight.copy_(weight)
    for p in conv.parameters():
        p.requires_grad = False
    return conv

class ARTBlock(nn.Module):
    def __init__(self, dim):
        super(ARTBlock, self).__init__()

        # fixed filters
        self.blur = AOTfilter(dim, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], norm=16.0)
        self.sobelx = AOTfilter(dim, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobely = AOTfilter(dim, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # task specific branches
        self.edge = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True)
        )
        self.low = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True)
        )
        self.mid = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.high = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

        # learnable branch weights
        self.alpha = nn.Parameter(torch.zeros(1, 4, 1, 1))

    @torch.no_grad()
    def reset(self):
        device = self.blur.weight.device
        dtype = self.blur.weight.dtype
        blur = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype, device=device) / 16.0
        sobelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
        sobely = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
        in_channels = self.blur.in_channels

        self.blur.weight.copy_(blur.view(1, 1, 3, 3).expand(in_channels, 1, 3, 3))
        self.sobelx.weight.copy_(sobelx.view(1, 1, 3, 3).expand(in_channels, 1, 3, 3))
        self.sobely.weight.copy_(sobely.view(1, 1, 3, 3).expand(in_channels, 1, 3, 3))

    def _feat_mask(self, mask, dims):
        """Downscale mask to feature dimensions."""
        B, C, H, W = mask.shape
        fH, fW = dims
        kH, kW = max(H // fH, 1), max(W // fW, 1)
        feat_mask = F.avg_pool2d(mask.float(), (kH, kW), (kH, kW))
        feat_mask = (feat_mask > 0.5).float()[:, :, :fH, :fW]
        feat_mask = F.avg_pool2d(feat_mask, 3, 1, 1).clamp_(0, 1)
        return feat_mask

    def forward(self, x, mask=None):
        if mask is not None:
            band = self._feat_mask(mask, x.shape[-2:])
        else:
            band = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)

        blur = self.blur(x)
        low = self.low(blur) # palette / smooth colour
        high = self.high(x - blur) # texture / brush

        gx = self.sobelx(x)
        gy = self.sobely(x)
        edge = self.edge(torch.abs(gx) + torch.abs(gy))
        mid = self.mid(x)  # mid structure

        scale = 1.0 + 0.25 * torch.tanh(self.alpha) # range [0.75, 1.25]
        edge = edge * scale[:,0:1]
        low = low * scale[:,1:2]
        mid = mid * scale[:,2:3]
        high = high * scale[:,3:4]

        out = self.fuse(torch.cat([edge, low, mid, high], dim=1))
        gate = torch.sigmoid(aot_layer_norm(self.gate(x)))
        gated = gate * band
        return x * (1 - gated) + out * gated

