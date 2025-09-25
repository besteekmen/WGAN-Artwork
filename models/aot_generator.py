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
        self.decoder2 = nn.Sequential(
            UpConv(G_HIDDEN * 4, G_HIDDEN * 2),
            nn.ReLU(inplace=True),
            UpConv(G_HIDDEN * 2, G_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Conv2d(G_HIDDEN,3,3,stride=1, padding=1)
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

    def forward(self, x, mask):
        x = torch.cat((x * (1.0 - mask), mask), dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.tanh(x)

def aot_layer_norm(features):
    mean = features.mean((2, 3), keepdim=True)
    std = features.std((2, 3), keepdim=True) + 1e-9
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

    def forward(self, x):
        out = torch.cat([
            self.block0(x), self.block1(x), self.block2(x), self.block3(x)
        ], dim=1)
        out = self.fuse(out)
        mask = torch.sigmoid(aot_layer_norm(self.gate(x)))
        return x * (1 - mask) + out * mask

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv(x)

