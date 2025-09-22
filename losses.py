import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from config import SCALES, HOLE_LAMBDA, VALID_LAMBDA, EPS, EDGE_RING, LPIPS_RING, VGG_RING
from utils.utils import get_device
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.vision_utils import downsample

def init_losses(device):
    """Initialize the losses for model networks."""
    lossStyle = VGG19StyleLoss().to(device)
    lossPerceptual = VGG16PerceptualLoss().to(device)
    lossLPIPS = VGGLPIPSLoss().to(device)
    return lossStyle, lossPerceptual, lossLPIPS

def init_metrics(device):
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
        # TODO: try below (and the one in VGG16) for offline, and weights is preferred over pretrained now by torchvision
        #vgg = tvmodels.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
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
        for idx, layer in enumerate(self.vgg):
            r = layer(r)
            f = layer(f)
            if idx in self.layers:
                real_features.append(r)
                fake_features.append(f)

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
        #vgg = tvmodels.vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
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
        for idx, layer in enumerate(self.vgg):
            r = layer(r)
            f = layer(f)
            if idx in self.layers:
                real_features.append(r)
                fake_features.append(f)

        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += self.criterion(rf, ff)
        return loss

class VGGLPIPSLoss(nn.Module):
    """Compute LPIPS on a ring around the boundary (frozen).
    Attributes:
    """
    def __init__(self, ring_type="outer", size=LPIPS_RING):
        super().__init__()
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").eval()
        for param in lpips.parameters():
            param.requires_grad = False
        self.lpips = lpips
        self.ring_type = ring_type
        self.size = size

    def forward(self, real, fake, mask_hole):
        ring = get_ring(mask_hole, self.size)[self.ring_type].to(fake.dtype).float()

        real = (real + 1.0) / 2.0
        fake = (fake + 1.0) / 2.0

        real = real * ring + EPS
        fake = fake * ring + EPS

        return self.lpips(real, fake).mean()

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
    diff = (x - y).abs()
    # normalize by mask area for stability (mask size invariant, per pixel error)
    # [B, 1, H, W] -> [B, 1*H*W] -> [B] by flatten and sum -> scalar by mean
    num = (diff * mask).flatten(1).sum(1)
    denom = torch.clamp(mask.flatten(1).sum(1) + EPS, min=1.0)
    return (num / denom).mean()

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
    multi_loss = 0
    r_scale = downsample(real, SCALES)
    f_scale = downsample(fake, SCALES)
    m_scale = downsample(mask, SCALES)
    for ors, fs, ms in zip(r_scale, f_scale, m_scale):
        hole = masked_l1(fs, ors, ms)
        valid = masked_l1(fs, ors, 1.0 - ms)
        multi_loss += HOLE_LAMBDA * hole + VALID_LAMBDA * valid
    return multi_loss / len(SCALES)

def sobel(x):
    """Applies sobel edge detector to input image."""
    x_gray = x.mean(dim=1, keepdim=True) # convert [B, 3, H, W] in [-1, 1] to grayscale
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=x.dtype,
        device=x.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=x.dtype,
        device=x.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(x_gray, sobel_x, padding=1)
    grad_y = F.conv2d(x_gray, sobel_y, padding=1)
    return torch.sqrt(grad_x.float() ** 2 + grad_y.float() ** 2 + EPS).to(x.dtype) # added epsilon to avoid NaN grads

def dilation(x, size=3):
    # x = [B, 1, H, W] in [0, 1] i.e. mask_hole so mask 1, rest 0
    return F.max_pool2d(x, kernel_size=(2 * size + 1), stride=1, padding=size)

def erosion(x, size=3):
    return 1.0 - F.max_pool2d(1.0 - x, kernel_size=(2 * size + 1), stride=1, padding=size)

def get_ring(x, size=3):
    dil = dilation(x, size)
    er = erosion(x, size)
    ring = {
        "inner": torch.clamp(x - er, 0.0, 1.0),
        "outer": torch.clamp(dil - x, 0.0, 1.0),
        "both": torch.clamp(dil - er, 0.0, 1.0)
    }
    return ring

def lossEdge(real, fake):
    return F.l1_loss(sobel(fake), sobel(real), reduction="mean") # use functional l1, not class one

def lossEdgeRing(real, fake, mask_hole, size=EDGE_RING, ring_type="both"):
    ring = get_ring(mask_hole, size)[ring_type].to(fake.dtype).float()
    return masked_l1(sobel(fake), sobel(real), ring)

def lossVGGRing(module, real, fake, mask_hole, size=VGG_RING, ring_type="outer"):
    ring = get_ring(mask_hole, size)[ring_type].to(fake.dtype).float()
    r = real * ring + EPS
    f = fake * ring + EPS
    return module(r, f) * vgg_scale(ring)

def lossTV(x, mask):
    """Return Total Variation (how much neighbours change).
    Calculate over the hole only, anisotropic so preserve edges."""
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    if mask is not None:
        my = mask[:, :, 1:, :]
        mx = mask[:, :, :, 1:]
        dy = dy * my
        dx = dx * mx
        num = (dy.sum() + dx.sum())
        denom = (my.sum() + mx.sum()).clamp_min(1.0)
        return num / denom
    return dy.mean() + dx.mean()

def vgg_scale(mask):
    """Approximate per-pixel average for various mask holes."""
    B, C, H, W = mask.size()
    total = float(H * W)
    active = mask.sum(dim=(1, 2, 3)).mean()
    return torch.clamp(total / (active + EPS), max=2.0)

