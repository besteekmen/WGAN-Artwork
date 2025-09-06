import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.backends import cudnn

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.generator import Generator
from models.weights_init import weights_init_normal
from data_utils import CroppedImageDataset, make_dataloader
from utils import to_unit, to_signed, clear_folder
from train import VGGFeatureExtractor, feature_loss, gram_matrix
from config import *

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ---------------------
# Evaluation function
# ---------------------
def evaluate(model_path, dataset_path, save_dir=EVAL_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    # ---------------------
    # Setup
    # ---------------------
    seed_val = np.random.randint(1, 10000) if SEED is None else SEED
    print(f"Random Seed: {seed_val}")
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    cuda_available = CUDA and torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.manual_seed(seed_val)
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if cuda_available else "CPU only\n")

    cudnn.benchmark = True
    clear_folder(save_dir)

    # ---------------------
    # Trained generator
    # ---------------------
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    # ---------------------
    # Test dataset and loader
    # ---------------------
    test_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'test'), split='test')
    test_loader = make_dataloader(test_set, 'test', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available)

    # ---------------------
    # VGG feature extractors
    # ---------------------
    style_extractor = VGGFeatureExtractor(mode="style").to(device).eval()
    perceptual_extractor = VGGFeatureExtractor(mode="perceptual").to(device).eval()

    # ---------------------
    # Metrics
    # ---------------------
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    criterion = nn.MSELoss()
    ssim_scores, lpips_scores, style_scores, perc_scores = [], [], [], []

    # Evaluation loop
    with torch.no_grad():
        for i, (masked, original, mask_known) in enumerate(tqdm(test_loader, desc="Evaluating")):
            masked = masked.to(device)
            original = original.to(device)
            mask_known = mask_known.to(device)
            mask_hole = (1.0 - mask_known).float().to(device)

            # Generator forward w/o grad
            fake, _ = netG(original, mask_hole)
            composite = fake * mask_hole + original * (1.0 - mask_hole)

            # Convert to [0,1] for metrics
            unit_original = to_unit(original)
            unit_comp = to_unit(composite)

            # SSIM and LPIPS metrics
            ssim_scores.append(ssim(unit_comp, unit_original).item())
            lpips_scores.append(lpips(unit_comp, unit_original).item())

            # Style and perceptual
            real_style = style_extractor(original)
            fake_style = style_extractor(composite)
            sl = feature_loss(real_style, fake_style, "style", criterion)
            style_scores.append(sl)

            real_perc = perceptual_extractor(original)
            fake_perc = perceptual_extractor(composite)
            pl = feature_loss(real_perc, fake_perc, "perceptual", criterion)
            perc_scores.append(pl)

            # Save images
            grid = torch.cat(
                [unit_original, to_unit(masked), unit_comp],
                3)  # horizontal stack per sample using width dimension
            vutils.save_image(grid,
                              os.path.join(save_dir, f'sample({i}).png'),
                              normalize=True,
                              nrow=1)  # 1 row per sample (not BATCH_SIZE)?

    # ---------------------
    # Aggregate metrics
    # ---------------------
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Avg LPIPS: {np.mean(lpips_scores):.4f}")
    print(f"Avg Style Loss: {np.mean(perc_scores):.4f}")
    print(f"Avg Perceptual Loss: {np.mean(style_scores):.4f}")

if __name__ == '__main__':
    test_dir = os.path.join(DATA_PATH, 'test')
    model = os.path.join(OUT_PATH, 'netG_epoch40.pth')
    evaluate(model, test_dir)