import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from config import SCALES, HOLE_LAMBDA, VALID_LAMBDA, EPS, TV_RING
from utils.utils import get_device, to_unit
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.vision_utils import downsample, get_ring

def init_losses(device=get_device()):
    """Initialize the losses for model networks."""
    lossStyle = VGG19StyleLoss().to(device)
    lossPerceptual = VGG16PerceptualLoss().to(device)
    return lossStyle, lossPerceptual

def init_metrics(device=get_device()):
    """Initialize the metrics for model networks."""
    # no need to normalize as unit ones are fed!
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    return ssim, lpips, fid

class VGG19StyleLoss(nn.Module):
    """Extract feature maps for style loss (frozen).

    Gets intermediate feature maps from VGG19,
    then calculates style loss between image feature maps.
    Default layers for style gram: (relu1_1, relu2_1, relu3_1, relu4_1)

    Attributes:
        layers: A list indicating which layers to use for style Gram

    """
    def __init__(self, layers=None):
        super().__init__()
        vgg = tvmodels.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = [int(x) for x in (layers or [1, 6, 11, 20])]
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.MSELoss()

    @staticmethod
    def gram(features: torch.Tensor) -> torch.Tensor:
        """Gram matrix for style loss.

        Calculates and returns gram matrix with shape [B, C, C]
        Attributes:
            features: A tensor of shape [B, C, H, W]
        """
        B, C, H, W = features.size()
        feats = features.view(B, C, H * W)
        denominator = (C * H * W) + EPS # not (H * W) to dampen style dominance
        return torch.bmm(feats, feats.transpose(1, 2)) / denominator

    def forward(self, real, fake):
        """
        real, fake: [B, 3, H, W] values in [-1, 1]
        """
        real = (real + 1.0) / 2.0 # try using to_unit here!
        fake = (fake + 1.0) / 2.0
        real = (real - self.mean) / self.std
        fake = (fake - self.mean) / self.std

        real_features = []
        fake_features = []
        r = real
        f = fake
        max_idx = max(self.layers)
        for idx, layer in enumerate(self.vgg):
            r = layer(r)
            f = layer(f)
            if idx in self.layers:
                real_features.append(r)
                fake_features.append(f)
            if idx >= max_idx:
                break

        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += self.criterion(self.gram(rf), self.gram(ff))
        return loss

class VGG16PerceptualLoss(nn.Module):
    """Extract feature maps for perceptual loss (frozen).

    Gets intermediate feature maps from VGG16,
    then calculates perceptual loss between image feature maps.
    Helps generator to match semantic structures.
    Default layers for perceptual loss: (relu1_2, relu2_2, relu3_3, relu4_3)

    Attributes:
        layers: A list of integers indicating which layers to use
    """
    def __init__(self, layers=None, resize=True):
        super().__init__()
        vgg = tvmodels.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = [int(x) for x in (layers or [3, 8, 15, 22])]
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.MSELoss()

    def forward(self, real, fake):
        real = (real + 1.0) / 2.0
        fake = (fake + 1.0) / 2.0

        real = (real - self.mean) / self.std
        fake = (fake - self.mean) / self.std

        real_features = []
        fake_features = []
        r = real
        f = fake
        max_idx = max(self.layers)
        for idx, layer in enumerate(self.vgg):
            r = layer(r)
            f = layer(f)
            if idx in self.layers:
                real_features.append(r)
                fake_features.append(f)
            if idx >= max_idx:
                break

        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += self.criterion(rf, ff)
        return loss

def gradient_penalty(critic, real, fake, device):
    """Return gradient penalty for a given gradient
    Src: https://medium.com/@krushnakr9/gans-wasserstein-gan-with-gradient-penalty-wgan-gp-b8da816cb2d2"""
    B, C, H, W = real.shape
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = alpha * real + ((1 - alpha) * fake)
    interpolated.requires_grad_(True)

    critic_scores = critic(interpolated)
    # average to scalar per image for WGAN-GP stability
    if critic_scores.dim() > 1:
        critic_scores = critic_scores.mean(dim=1)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        only_inputs=True
        #retain_graph=True # removed since it was not necessary
    )[0]

    gradients = gradients.view(B, -1)
    grad_norm = torch.clamp(gradients.norm(2, dim=1), 0, 10) + EPS # stabilizer and clamp added to avoid NaN g loss
    return ((grad_norm - 1) ** 2).mean()

def masked_l1(x, y, mask):
    """Mean |x-y| over elements where mask==1, normalized by the number
    of selected elements (including channels), averaged over batch
    """
    diff = (x - y).abs() # [B,C,H,W]
    # normalize by mask area for stability (mask size invariant)
    num = (diff * mask).sum(dim=(1, 2, 3)) # [B]
    denom = (mask.sum(dim=(1, 2, 3)) * x.size(1)).clamp_min(EPS) # [B]
    return (num / denom).mean() # scalar

def lossMSL1(real, fake, mask):
    """Calculate multiscale loss for a given loss function.

    Pixel-wise loss is calculated for the whole image at different scales,
    as tracking only the hole cause seam artifacts at boundary and inconsistencies
    Weighted l1 loss: https://arxiv.org/pdf/2401.03395
    Also weighted: https://arxiv.org/pdf/1801.07892

    Arguments:
        real: [B, 3, H, W] values in [-1, 1]
        fake: [B, 3, H, W] values in [-1, 1]
        mask: [B, 1, H, W] values in [0, 1], (1=hole, 0=known)
    """
    frac_hole = mask.mean().detach() # tensor scalar on CUDA
    frac_valid = 1.0 - frac_hole

    r_scale = downsample(real, SCALES)
    f_scale = downsample(fake, SCALES)
    m_scale = downsample(mask, SCALES)
    multi_loss = 0

    #multi_loss = real.new_tensor(0.0) # GPU scalar (no switch to CPU)
    for ors, fs, ms in zip(r_scale, f_scale, m_scale):
        ms = ms.clamp(0.0, 1.0)
        hole = masked_l1(fs, ors, ms)
        valid = masked_l1(fs, ors, 1.0 - ms)
        multi_loss += HOLE_LAMBDA * hole * frac_hole + VALID_LAMBDA * valid * frac_valid
    return multi_loss / len(SCALES)

def sobel(x):
    """Applies sobel edge detector to input image."""
    x_gray = x.mean(dim=1, keepdim=True) # convert [B, 3, H, W] in [-1, 1] to grayscale
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=torch.float32,
        device=x.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=torch.float32,
        device=x.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(x_gray, sobel_x, padding=1)
    grad_y = F.conv2d(x_gray, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + EPS) # added epsilon to avoid NaN grads

def lossEdge(real, fake):
    return F.l1_loss(sobel(real), sobel(fake)) # use functional l1, not class one

def lossTV(x, mask, size=TV_RING):
    """Return Total Variation (how much neighbours change).
    Calculate over the ring only, anisotropic so preserve edges."""
    ring = get_ring(mask, size, blur_kernel=5, normalize=False)["both"].to(mask.dtype)

    # finite differences
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()

    # crop to match shape
    ringx = (ring[:, :, :, 1:] * ring[:, :, :, :-1]).to(x.dtype)
    ringy = (ring[:, :, 1:, :] * ring[:, :, :-1, :]).to(x.dtype)

    tvx = (dx.abs() * ringx).sum()
    tvy = (dy.abs() * ringy).sum()
    denom = (ringx.sum() + ringy.sum()).clamp_min(1.0)
    return (tvx + tvy) / denom

def lossFM(real_feats, fake_feats):
    fm = 0.0
    for ff, rf in zip(real_feats, fake_feats):
        fm += (ff - rf).abs().mean()
    return fm / len(fake_feats)