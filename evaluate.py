import os
from tqdm import tqdm

import torch
from torchvision import transforms
import torchvision.utils as vutils

from models.aot_generator import AOTGenerator
from dataset import CroppedImageDataset, randomize_masks, make_dataloader
from utils.utils import to_unit, print_device, set_seed, get_device, is_cuda, to_u8
from config import NUM_WORKERS, SEED, BATCH_SIZE, DATA_PATH

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def load_state(netG, path, device, strict=False):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "ema", "generator", "netG", "G", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    missing, unexpected = netG.load_state_dict(checkpoint, strict=strict)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

def evaluate(model_path, out_dir="eval_outputs",
             irr_ratio=0.3, batch_size=BATCH_SIZE,
             save_images=True):
    print_device()
    set_seed(SEED)
    device = get_device()

    # Load generator
    netG = AOTGenerator(in_channels=4).to(device).eval()
    load_state(netG, model_path, device)

    # Load validation dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'test'), transform=transform, split='test')
    test_loader = make_dataloader(test_set, 'test', batch_size, num_workers=NUM_WORKERS, cuda=is_cuda(), shuffle=False)

    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.reset()

    if save_images:
        os.makedirs(out_dir, exist_ok=True)

    ssim_tot, lpips_tot, batch = 0.0, 0.0, 0
    counter = 0

    # Evaluation loop
    with torch.inference_mode():
        for image, mask in tqdm(test_loader, desc="Testing", ncols=100):
            image = image.to(device, non_blocking=True)
            mask_known = randomize_masks(mask.to(device), irr_ratio=irr_ratio)
            mask_hole = (1.0 - mask_known).float()

            fake = netG(image, mask_hole)
            masked = image * mask_known
            comp = masked + fake * mask_hole

            unit_image = to_unit(image) # [0,1]
            unit_comp = to_unit(comp) # [0,1]

            ssim_batch = ssim(unit_comp, unit_image).item()
            ssim_tot += ssim_batch

            lpips_batch = lpips(unit_image, unit_comp).item()
            lpips_tot += lpips_batch

            fid.update(to_u8(unit_image), real=True)
            fid.update(to_u8(unit_comp), real=False)

            if save_images:
                grid = torch.cat(  # dimensions: (C,H,W)
                    [unit_image, to_unit(masked), unit_comp], 3)
                vutils.save_image(
                    grid,
                    os.path.join(out_dir, f'test_image({counter}).jpg'),
                    normalize=False,
                    nrow=1
                )
                counter += 1
            batch += 1

    ssim_avg = ssim_tot / max(1, batch)
    lpips_avg = lpips_tot / max(1, batch)
    fid_score = fid.compute().item()

    print(f"[Test] SSIM: {ssim_avg:.4f} | LPIPS: {lpips_avg:.4f} | FID: {fid_score:.4f}")
    return ssim_avg, lpips_avg, fid_score

if __name__ == '__main__':
    evaluate("netG_ema_final.pt")