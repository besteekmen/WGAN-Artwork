import torch
import torch.nn as nn
import torchvision.models as tvmodels
from config import *

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
