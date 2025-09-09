import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from models.generator import Generator
from dataset import CroppedImageDataset
from utils import to_unit, to_signed
from config import *

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def evaluate(model_path, dataset_path, results_file="evaluation_results.txt", save_images=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    # Load validation dataset
    transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
    ])
    val_dataset = CroppedImageDataset(root_dir=dataset_path, transform=transform, mask_type="random")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(normalize=True).to(device)

    ssim_scores, lpips_scores = [], []

    save_dir = "eval_outputs"
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    # Evaluation loop
    with torch.no_grad():


if __name__ == '__main__':
    generate('output/net_g_10.pth', device='cuda' if CUDA and torch.cuda.is_available() else 'cpu')