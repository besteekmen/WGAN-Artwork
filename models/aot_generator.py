import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.vision_utils import get_ring
from models.weights_init import weights_init_normal
from config import *

# --------------
# AOT Generator: switched from coarse to fine!
# --------------
class AOTGenerator(nn.Module):
    def __init__(self, in_channels=4):
        super(AOTGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            # 1st layer (Input: 3 x 262 x 262 -> 256 + 6 by padding)
            nn.Conv2d(in_channels, G_HIDDEN, 7), # [B, 64, 256, 256]
            nn.ReLU(inplace=True),
            # inplace ReLU is used to prevent 'Out of memory'
            # Why?: https://discuss.pytorch.org/t/guidelines-for-when-and-why-one-should-set-inplace-true/50923
            # 2nd layer
            nn.Conv2d(G_HIDDEN, G_HIDDEN * 2, 4, stride=2, padding=1),  # [B, 128, 128, 128]
            nn.ReLU(inplace=True),
            # 3rd layer
            nn.Conv2d(G_HIDDEN * 2, G_HIDDEN * 4, 4, stride=2, padding=1),  # [B, 256, 64, 64]
            nn.ReLU(inplace=True)
        )

        self.aot1 = AOTBlock(G_HIDDEN * 4)
        #self.aot2 = AOTBlock(G_HIDDEN * 4) # If you use it, also edit forward function!
        self.aot3 = AOTBlock(G_HIDDEN * 4)
        self.aot4 = AOTBlock(G_HIDDEN * 4)

        self.deconv1 = nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, stride=2, padding=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, stride=2, padding=1, bias=True)
        self.to_rgb = nn.Conv2d(G_HIDDEN, 3, 3, padding=1)

        self.apply(weights_init_normal)

        self.blur_up1 = AOTfilter(G_HIDDEN * 2, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], norm=16.0)
        self.blur_up2 = AOTfilter(G_HIDDEN, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], norm=16.0)

        # Low resolution RGB: bottleneck to rgb
        self.mid_to_rgb = nn.Sequential(
            nn.Conv2d(G_HIDDEN * 4, G_HIDDEN, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(G_HIDDEN, 3, 3, padding=1),
            nn.Tanh()
        )
        self.rgb_to_mid = nn.Conv2d(3, G_HIDDEN * 4, 3, padding=1)
        self.ced_param = CEDParamHead(in_ch=4, hidden=32)
        self.ced1 = CEDStep(iters=1, step_clip=0.08, k_sigma=3, k_rho=11)
        self.ced2 = CEDStep(iters=2, step_clip=0.08, k_sigma=3, k_rho=13)

        self.mid_to_rgb.apply(weights_init_normal)
        self.rgb_to_mid.apply(weights_init_normal)
        self.ced_param.apply(weights_init_normal)

    def forward(self, x, mask):
        """Gets an image and a mask to forward. Original image is fed,
        and the masked image is generated inside. (but feeding only masked!)

        Arguments:
            x: [B, 3, H, W] values in [-1, 1]
            mask: [B, 1, H, W] values in [0, 1], (1=hole, 0=known)
        """
        masked_input = x * (1.0 - mask)
        # TODO: Update later and do masking outside, it is confusing! :P

        x = torch.cat((masked_input, mask), dim=1)
        x = self.encoder(x)
        x = self.aot1(x)
        #x = self.aot2(x) # epoch time increases! But could be added later.

        # ----------- Diffusion Part -----------
        #img_pred = self.mid_to_rgb(x) # [B,3,64,64]

        # Option 1: if interpolate rather than pooling
        #img_ctx = F.interpolate(masked_input, size=img_pred.shape[-2:], mode='bilinear', align_corners=False)
        #m = F.interpolate(mask, size=img_pred.shape[-2:], mode='nearest').clamp(0,1)

        # Option 2: if pooling rather than interpolate (Faster)
        #s = masked_input.shape[-1] // img_pred.shape[-1]
        #img_ctx = F.avg_pool2d(masked_input, kernel_size=s, stride=s)
        #m = F.max_pool2d(mask, kernel_size=s, stride=s).clamp(min=0, max=1)

        #img_comp = img_ctx * (1 - m) + img_pred * m
        #tau, alpha, C, gamma = self.ced_param(torch.cat([img_comp, m], dim=1))
        #img_ced = self.ced(img_comp, img_ctx, m, tau=tau, alpha=alpha, C=C)
        #delta_feat = self.rgb_to_mid(img_ced - img_pred) # [B,G_HIDDEN * 4,64,64]
        #x = x + gamma * delta_feat
        # ----------- End of Diffusion ---------

        # ----------- Diffusion 2 CED ----------
        # ----------- CED cache (once) ---------
        Hf, Wf = x.shape[-2], x.shape[-1] # feature grid (64x64)
        #s = masked_input.shape[-1] // Wf # 256//64 = 4

        img_ctx = F.interpolate(masked_input, size=(Hf,Wf), mode="bilinear", align_corners=False) # [B,3,Hf,Wf]
        m = F.interpolate(mask, size=(Hf,Wf), mode="nearest").clamp_(0,1) # [B,1,Hf,Wf]

        def ced_inject(feat, ced, gamma_scale=1.0, tau_scale=1.0):
            img_pred = self.mid_to_rgb(feat)
            img_comp = img_ctx * (1.0 - m) + img_pred * m
            tau, alpha, C, gamma = self.ced_param(torch.cat([img_comp, m], dim=1))

            tau = tau * tau_scale

            img_ced = ced(img_comp, img_ctx, m, tau=tau, alpha=alpha, C=C)
            delta_feat = self.rgb_to_mid(img_ced - img_pred)
            return feat + (gamma_scale * gamma) * delta_feat

        # ----------- CED pass 1 (after aot1) --
        x = ced_inject(x, self.ced1, gamma_scale=0.45, tau_scale=0.8)

        # ----------- End of Df 2 CED ----------

        x = self.aot3(x)
        # ----------- CED pass 2 (after aot3) --
        x = ced_inject(x, self.ced2, gamma_scale=0.2, tau_scale=1.0)
        x = self.aot4(x)

        # decoder part
        x = self.deconv1(x)
        x = self.blur_up1(x)
        x = F.relu(x, inplace=True)
        x = self.deconv2(x)
        x = self.blur_up2(x)
        x = F.relu(x, inplace=True)
        x = self.to_rgb(x)
        return torch.tanh(x)

def aot_layer_norm(features):
    """Normalizes AOT block after the gating and before the sigmoid."""
    mean = features.mean((2, 3), keepdim=True)
    std = features.std((2, 3), keepdim=True) + 1e-9 # Try global eps here.
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

        # learnable branch weights, lets adaptable contribution of different receptive fields
        self.alpha = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, x):
        """Multiple receptive field convolutions for AOT generator.
        Gets the feature map, forwards it through 4 different dilation branches,
        scale the outputs with a learnable parameter (how much each detailedness will contribute to proposal),
        concatenates each C_bottleneck/4 scaled output and fuses them to C_bottleneck channel,
        passes through a gate to decide where the proposals be applied (per pixel),
        then normalizes and applies sigmoid.
        The output is like masking with residual, which applies the changes only at
        decided places

        Arguments:
            x: [B, G_HIDDEN * 4, H/4, W/4] feature maps
        """
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
        g = torch.sigmoid(aot_layer_norm(self.gate(x)))
        return x * (1 - g) + out * g

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

class CEDParamHead(nn.Module):
    def __init__(self, in_ch=4, hidden=32):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 4, kernel_size=1) # [B,4,1,1]

    def forward(self, x):
        feat = self.feat(x) # [B,hidden,H,W]
        m = x[:, -1:, :, :] # [B,1,H,W] (1=hole)

        w = m # focus on hole pixels
        wsum = w.sum(dim=(2,3), keepdim=True).clamp_min(1e-6)
        pooled = (feat * w).sum(dim=(2,3), keepdim=True) / wsum # [B,hidden,1,1]

        raw = self.out(pooled).view(x.size(0), 4) # [B,4]
        raw_tau, raw_alpha, raw_C, raw_gamma = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

        # set safe ranges
        tau = 0.02 + 0.18 * torch.sigmoid(raw_tau)
        alpha = 0.001 + 0.049 * torch.sigmoid(raw_alpha)
        C = 0.10 + 9.90 * torch.sigmoid(raw_C)
        gamma = 0.05 + 0.45 * torch.sigmoid(raw_gamma)

        # reshape to broadcast over H, W
        B = x.size(0)
        return (tau.view(B,1,1,1),
                alpha.view(B,1,1,1),
                C.view(B,1,1,1),
                gamma.view(B,1,1,1))

class CEDStep(nn.Module):
    """
    Coherence-Enhancing Diffusion (CED) step on RGB (or feature-like) images.

    - Structure-tensor orientation/coherence is computed from img_comp DETACHED.
    - tau/alpha/C keep gradient (they affect d_n, d_t and explicit update.)
    - Tensor math is in float32 even under autocast, then back at the end.
    """
    def __init__(self, iters=2, step_clip=0.10, k_sigma=3, k_rho=13):
        super().__init__()
        self.iters = int(iters)
        self.step_clip = float(step_clip)
        self.k_sigma = int(k_sigma) # pre-smooth before gradient
        self.k_rho = int(k_rho) # tensor integration scale

        # Central difference kernels (h=1 on current grid)
        kx = torch.tensor([[0,0,0],
                           [-0.5,0,0.5],
                           [0,0,0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[0, -0.5,0],
                           [0,0,0],
                           [0,0.5,0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _dw(self, x, k):
        C = x.size(1)
        w = k.expand(C, 1, 3, 3)
        return F.conv2d(x, w, padding=1, groups=C)

    def _blur(self, x, k):
        pad = k // 2
        return F.avg_pool2d(x, k, 1, pad)

    def forward(self, img_comp, img_ctx, m, tau, alpha, C):
        """
        All tensors are [B,*,H,W], params are [B,1,1,1]
        img_comp, img_ctx: [B,3,H,W] (or any C)
        m: [B,1,H,W]
        tau, alpha, C: [B,1,1,1]
        """
        eps = 1e-6
        out_dtype = img_comp.dtype

        # Build structure tensor from luminance
        u0 = img_comp.detach().float()
        if u0.size(1) == 3:
            g = 0.299*u0[:,0:1] + 0.587*u0[:,1:2] + 0.114*u0[:,2:3]
        else:
            g = u0.mean(dim=1, keepdim=True)

        # sigma: smooth before gradient
        g = self._blur(g, self.k_sigma)

        gx = F.conv2d(g, self.kx, padding=1)
        gy = F.conv2d(g, self.ky, padding=1)

        # rho: integrate tensor over larger neighbourhood, normalize by known pixel support
        w = (1.0 - m.float()) # known=1, hole=0
        wr = self._blur(w, self.k_rho).clamp_min(1e-6)

        J11 = self._blur((gx * gx) * w, self.k_rho) / wr
        J22 = self._blur((gy * gy) * w, self.k_rho) / wr
        J12 = self._blur((gx * gy) * w, self.k_rho) / wr

        # Dominant eigenvector angle (normal direction)
        theta = 0.5 * torch.atan2(2.0 * J12, (J11 - J22) + eps)

        # Coherence measure from eigenvalues
        tr = J11 + J22
        det = torch.sqrt((J11 - J22)**2 + 4.0 * (J12**2) + eps)
        mu1 = 0.5 * (tr + det)
        mu2 = 0.5 * (tr - det)
        coh = (mu1 - mu2)**2 # coherence measure (detached wrt img_comp)

        # diffusion coefficients (KEEP grad wrt alpha, C)
        alpha_f = alpha.float()
        C_f = C.float()
        tau_f = tau.float()
        # across-edge (normal) = alpha (small)
        d_n = alpha_f
        # along-edge (tangent) grows toward 1 when coherence is high
        d_t = alpha_f + (1.0 - alpha_f) * torch.exp(-C_f / (coh + eps))

        c = torch.cos(theta)
        s = torch.sin(theta)
        # v_n = (c,s) and v_t = (-s,c)
        vnx, vny = c, s
        vtx, vty = -s, c

        D11 = d_n * (vnx*vnx) + d_t * (vtx*vtx)
        D22 = d_n * (vny*vny) + d_t * (vty*vty)
        D12 = d_n * (vnx*vny) + d_t * (vtx*vty)

        # Explicit update in float32, mask-only, then enforce context
        u = img_comp.float()
        ctx = img_ctx.float()
        m_soft = m.float()
        m_hard = (m_soft > 0.5).float()

        for _ in range(self.iters):
            ux = self._dw(u, self.kx)
            uy = self._dw(u, self.ky)

            px = D11 * ux + D12 * uy
            py = D12 * ux + D22 * uy

            div = self._dw(px, self.kx) + self._dw(py, self.ky)

            step = (tau_f * div).clamp(-self.step_clip, self.step_clip)
            u = u + step * m_soft
            u = u * m_hard + ctx * (1.0 - m_hard)
        return u.to(out_dtype)
