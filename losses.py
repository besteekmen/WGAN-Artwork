import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from config import SCALES, HOLE_LAMBDA, VALID_LAMBDA
from utils.utils import get_device
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.vision_utils import downsample

def init_losses(device=get_device()):
    """Initialize the losses for model networks."""
    lossStyle = VGG19StyleLoss().to(device)
    lossPerceptual = VGG16PerceptualLoss().to(device)
    return lossStyle, lossPerceptual

def init_metrics(device=get_device()):
    """Initialize the metrics for model networks."""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
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
        denominator = (C * H * W) + 1e-8
        return torch.bmm(feats, feats.transpose(1, 2)) / denominator

    def forward(self, real, fake):
        """
        real, fake: [B, 3, H, W] values in [-1, 1]
        """
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

def gradient_penalty(critic, real, fake, device):
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
        only_inputs=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(B, -1)
    grad_norm = gradients.norm(2, dim=1) + 1e-8 # stabilizer added to avoid nan g loss TODO:check!
    return ((grad_norm - 1) ** 2).mean()

def lossMSL1(real, fake, mask):
    """Calculate multiscale loss for a given loss function.
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
        multi_loss += HOLE_LAMBDA * F.l1_loss(fs * ms, ors * ms) + \
                      VALID_LAMBDA * F.l1_loss(fs * (1.0 - ms), ors * (1.0 - ms))
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
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6) # added epsilon to avoid NaN grads

def lossEdge(real, fake):
    return F.l1_loss(sobel(real), sobel(fake)) # use functional l1, not class one
