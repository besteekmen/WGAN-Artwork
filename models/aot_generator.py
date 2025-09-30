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

        self.aot1 = AOTBlock(G_HIDDEN * 4)
        #self.aot2 = AOTBlock(G_HIDDEN * 4)
        #self.art = ARTBlock(G_HIDDEN * 4)
        self.ced = CEDBlock(G_HIDDEN * 4, detach_orientation=True)
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
            if isinstance(m, (ARTBlock, CEDBlock)):
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
        x = self.ced(x, mask)
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
        blur_conv = self.blur[1]
        sobelx_conv = self.sobelx[1]
        sobely_conv = self.sobely[1]

        device = blur_conv.weight.device
        dtype = blur_conv.weight.dtype
        in_channels = blur_conv.in_channels

        blur = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype, device=device) / 16.0
        sobelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
        sobely = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)

        blur_conv.weight.copy_(blur.view(1, 1, 3, 3).expand(in_channels, 1, 3, 3))
        sobelx_conv.weight.copy_(sobelx.view(1, 1, 3, 3).expand(in_channels, 1, 3, 3))
        sobely_conv.weight.copy_(sobely.view(1, 1, 3, 3).expand(in_channels, 1, 3, 3))

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

class CEDBlock(nn.Module):
    def __init__(self, dim, steps=1, tau=0.15, alpha=1e-3, C=0.05,
                 m=2, rho=2, detach_orientation=True):
        super().__init__()
        self.steps = steps # k, so total time: T = k * Δt
        self.tau = tau # explicit time step: Δt, better <0.25
        self.alpha = alpha # minimal across edge diffusivity
        self.C = C # contrast parameter, more along-edge diff. if large
        self.m = m # steepness of switch: 1 smoother 2 crisper
        self.rho = rho # integration scale rho >= 2sigma
        self.detach_orientation = detach_orientation # if True, dont backprop

        # Gradients and blur for stable orientation
        self.blur1 = AOTfilter(1, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], norm=16.0)
        self.blur = AOTfilter(dim, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], norm=16.0)
        self.gx = AOTfilter(dim, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.gy = AOTfilter(dim, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Second-derivative bases
        self.dxx = AOTfilter(dim, [[1, -2, 1], [2, -4, 2], [1, -2, 1]])
        self.dyy = AOTfilter(dim, [[1, 2, 1], [-2, -4, -2], [1, 2, 1]])
        self.dxy = AOTfilter(dim, [[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

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
        _load(self.blur1, [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]])
        _load(self.gx, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        _load(self.gy, [[-1, -2, -1], [0, 0, 0], [-1, 0, 1]])

        _load(self.dxx, [[1, -2, 1], [2, -4, 2], [1, -2, 1]])
        _load(self.dyy, [[1, 2, 1], [-2, -4, -2], [1, 2, 1]])
        _load(self.dxy, [[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    def _down_mask(self, mask, H, W):
        """Downscale mask to feature dimensions."""
        kH, kW = max(mask.size(2) // H, 1), max(mask.size(3) // W, 1)
        down_mask = F.avg_pool2d(mask.float(), (kH, kW), (kH, kW))
        down_mask = (down_mask > 0.5).float()[:, :, :H, :W]
        down_mask = 1.0 - F.avg_pool2d(1.0 - down_mask, 3, 1, 1)
        return down_mask.clamp_(0, 1)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        eps = 1e-6
        band = self._down_mask(mask, H, W) if mask is not None else None

        # 1) Structure tensor J = blur([gx;gy] [gx;gy]^T)
        x_sigma = self.blur(x)
        gx = self.gx(x_sigma) # [B, C, H, W]
        gy = self.gy(x_sigma)
        j11 = (gx * gx).sum(1, keepdim=True) # [B, 1, H, W]
        j12 = (gx * gy).sum(1, keepdim=True)
        j22 = (gy * gy).sum(1, keepdim=True)

        # integrate over ρ (rho) using repeated gaussian passes
        for _ in range(self.rho):
            j11, j12, j22 = self.blur1(j11), self.blur1(j12), self.blur1(j22)

        s = (j11 + j22).clamp_min(eps)
        j11, j12, j22 = j11 / s, j12 / s, j22 / s
        # due to high number of channels scaled

        # 2) Eigenvalues μ1>=μ2 and major eigenvector angle θ
        tmp = torch.sqrt(((j11 - j22) ** 2 + 4.0 * j12 ** 2).clamp_min(eps))
        mu1 = 0.5 * (j11 + j22 + tmp) # largest mu
        mu2 = 0.5 * (j11 + j22 - tmp)

        # Coherence measure (un-normalized) and Weickert's λ
        delta = (mu1 - mu2) # Δ = μ1 - μ2
        lambda_perp = self.alpha # λ⟂ = α (across coherent structures)
        lambda_tang = self.alpha + (1.0 - self.alpha) * torch.exp(
            -self.C / (delta.pow(2 * self.m) + eps)
        ) # λ∥ = α + (1 - α)exp(-C/(μ1 - μ2)^2) (along coherent structures)

        # Major eigenvector (normal to edges), tangent is its perpendicular
        theta = 0.5 * torch.atan2(2.0 * j12, (j22 - j11 + eps))
        vx, vy = torch.cos(theta), torch.sin(theta)
        if self.detach_orientation:
            vx = vx.detach()
            vy = vy.detach()
            #lambda_tang = lambda_tang.detach()

        # 3) Explicit Euler
        x_new = x
        for _ in range(max(self.steps, 1)):
            # directional derivatives
            dxx = self.dxx(x_new)
            dyy = self.dyy(x_new)
            dxy = self.dxy(x_new)

            # second derivatives along normal (v) and tangent (v⟂)
            # v = (cosθ, sinθ)
            # Along edge (tangent) is v⟂, across edge (normal) is v
            d2n = (vx * vx) * dxx + 2.0 * (vx * vy) * dxy + (vy * vy) * dyy # v^T H v (across edge)
            d2t = (vy * vy) * dxx - 2.0 * (vx * vy) * dxy + (vx * vx) * dyy # v⟂^T H v⟂ (along edge)
            step = self.tau * (lambda_tang * d2t + lambda_perp * d2n)

            if band is not None:
                step = step * band
            x_new = x_new + step
        return x_new
