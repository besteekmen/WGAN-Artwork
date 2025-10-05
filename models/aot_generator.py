import torch
import torch.nn as nn
import torch.nn.functional as F

from models.weights_init import weights_init_normal
from config import *
from utils.vision_utils import get_ring


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

        self.aot1 = AOTBlock(G_HIDDEN * 4)
        #self.aot2 = AOTBlock(G_HIDDEN * 4)
        self.dif = DIFBlock(G_HIDDEN * 4, detach_orientation=True)
        self.aot3 = AOTBlock(G_HIDDEN * 4)
        self.aot4 = AOTBlock(G_HIDDEN * 4)

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
            if isinstance(m, (DIFBlock)):
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
        x = self.aot1(x)
        #x = self.aot2(x)
        x = self.dif(x, mask)
        x = self.aot3(x)
        x = self.aot4(x)
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
            nn.ReflectionPad2d(8), # try 12 later!
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
    conv = nn.Conv2d(channel, channel, 3, padding=0, groups=channel, bias=False)
    pad = nn.ReflectionPad2d(1)
    with torch.no_grad():
        weight = k.view(1, 1, 3, 3).expand(channel, 1, 3, 3).clone()
        conv.weight.copy_(weight)
    for p in conv.parameters():
        p.requires_grad = False
    return nn.Sequential(pad, conv)

class DIFBlock(nn.Module):
    def __init__(self, dim, steps=1, tau=0.2, alpha=0.9, beta=0.10,
                 detach_orientation=False):
        super().__init__()
        self.steps = steps
        # base parameters
        self.tau0 = float(tau)
        self.alpha0 = float(alpha)  # gain for along-edge (tangent) curvature
        self.beta0 = float(beta)  # gain for across-edge (normal) curvature
        self.scale = 1.0
        self.ring = 1
        self.detach_orientation = detach_orientation

        # Gradients and blur for stable orientation
        self.blur = AOTfilter(dim, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], norm=16.0)
        self.gx = AOTfilter(dim, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.gy = AOTfilter(dim, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Second-derivative bases
        self.dxx = AOTfilter(dim, [[1, -2, 1], [2, -4, 2], [1, -2, 1]])
        self.dyy = AOTfilter(dim, [[1, 2, 1], [-2, -4, -2], [1, 2, 1]])
        self.dxy = AOTfilter(dim, [[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    @torch.no_grad()
    def set_schedule(self, scale: float = 1.0):
        self.scale = float(scale)

    @torch.no_grad()
    def reset(self):
        def _load(seq, k):
            conv = seq[1]
            C = conv.in_channels
            device = conv.weight.device
            dtype = conv.weight.dtype
            w = torch.tensor(k, device=device, dtype=dtype).view(1, 1, 3, 3).expand(C, 1, 3, 3)
            conv.weight.copy_(w)

        _load(self.blur, [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])
        _load(self.gx, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        _load(self.gy, [[-1, -2, -1], [0, 0, 0], [-1, 0, 1]])

        _load(self.dxx, [[1, -2, 1], [2, -4, 2], [1, -2, 1]])
        _load(self.dyy, [[1, 2, 1], [-2, -4, -2], [1, 2, 1]])
        _load(self.dxy, [[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    def _down_mask(self, mask, H, W):
        """Downscale mask to feature dimensions."""
        return F.interpolate(mask.float(), size=(H, W), mode='nearest').clamp_(0, 1)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        band = self._down_mask(mask, H, W) if mask is not None else None
        if band is None:
            band = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        inner = get_ring(band, size=self.ring, blur_kernel=9, normalize=True)["inner"].to(band.dtype)

        with torch.no_grad():
            ctx = x * (1.0 - band)
            gx = self.blur(self.gx(ctx))
            gy = self.blur(self.gy(ctx))
            mean_x = gx.mean(1, keepdim=True)
            mean_y = gy.mean(1, keepdim=True)

            eps = 1e-6 if x.dtype == torch.float32 else 1e-4
            magnitude = (mean_x**2 + mean_y**2).sqrt().clamp_min(eps)

            # gradient direction (vx, vy) -> normal to edge
            # (-vy, vx) -> tangent to edge
            vx, vy = mean_x / magnitude, mean_y / magnitude

            # smooth and renormalize orientation
            vx = F.avg_pool2d(vx, kernel_size=3, stride=1, padding=1)
            vy = F.avg_pool2d(vy, kernel_size=3, stride=1, padding=1)
            den = (vx*vx + vy*vy).sqrt().clamp_min(eps)
            vx = vx/den
            vy = vy/den

            if self.detach_orientation:
                vx = vx.detach()
                vy = vy.detach()

            c_par = torch.sigmoid(4.0 * magnitude) # ~1 at strong edges
            c_perp = 0.25 * (1.0 - c_par)

            # local CFL safety scaling for explicit step
            q = (self.alpha0 * c_par).abs() + (self.beta0 * c_perp).abs()
            safety = 0.33 # 1/3 for a 3x3 stencil
            tau_eff = (self.tau0 * self.scale) * torch.clamp(safety / (q + eps), max=1.0)

        x_new = x
        for _ in range(self.steps):
            dxx = self.dxx(x_new)
            dyy = self.dyy(x_new)
            dxy = self.dxy(x_new)

            vxvx = vx * vx
            vyvy = vy * vy
            vxvy = vx * vy

            d2n = vxvx * dxx + 2 * vxvy * dxy + vyvy * dyy # curvature along normal
            d2t = vyvy * dxx - 2 * vxvy * dxy + vxvx * dyy # curvature along tangent
            core = self.alpha0 * c_par * d2t + self.beta0 * c_perp * d2n
            step = (tau_eff * core * inner).clamp(-0.05, 0.05)
            x_new = x_new + step
        return x_new