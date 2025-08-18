import os
# import random (to not use np.random instead)
import sys
from os import environ

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datetime import datetime
from tqdm import tqdm
from multiprocessing import freeze_support

from config import *
from utils import clear_folder
from data_utils import CroppedImageDataset
from models.generator import Generator
from models.discriminator import GlobalDiscriminator, LocalDiscriminator
# TODO: Should I import global local coarse fine etc?
from models.weights_init import weights_init_normal
# TODO: Should I use this below:
#os.environ["PYTHONUNBUFFERED"] = "1"

# Optional config fallbacks
OUT_PATH = globals().get('OUT_PATH', 'out')
SAMPLE_SAVE_STEP = globals().get('SAMPLE_SAVE_STEP', 200) # batches
CHECKPOINT_EVERY = globals().get('CHECKPOINT_EVERY', 1) # epochs

# ---------------------
# Utilities: Gram matrix and VGG extractor for style loss
# ---------------------
def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Calculate Gram matrix for style loss.
    Input (features): tensor of shape [B, C, H, W]
    Output: tensor of shape [B, C, C] TODO: check!"""
    B, C, H, W = features.size()
    feats = features.view(B, C, H * W)
    return torch.bmm(feats, feats.transpose(1, 2)) / (C * H * W)

class VGG19StyleExtractor(nn.Module):
    """Extract intermediate feature maps from VGG19 for style loss (frozen)."""
    def __init__(self, device, layers=None):
        super().__init__()
        vgg = tvmodels.vgg19(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        # common relu layers to use for style Gram: relu1_1, relu2_1, relu3_1, relu4_1
        # module indices: relu1_1='1', relu2_1='6', relu3_1='11', relu4_1='20'
        self.layers = layers or ['1', '6', '11', '20']
        # store ImageNet mean/std statistics for normalization (could be done with register_buffer later)
        self.mean = torch.Tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, x):
        # Rescale from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Normalize with ImageNet statistics
        x = (x - self.mean) / self.std
        # Pass through VGG and extract chosen layers
        features = []
        cur = x
        for name, layer in self.vgg.named_children():
            cur = layer(cur)
            if name in self.layers:
                features.append(cur)
                if len(features) == len(self.layers):
                    break
        return features

def style_loss(real_features, fake_features, criterion):
    """Compute style loss between real and fake feature maps."""
    loss = 0
    for rf, ff in zip(real_features, fake_features):
        loss += criterion["mse"](gram_matrix(rf), gram_matrix(ff))
    return loss

# ---------------------
# Helper: Crop a local square patch centered on mask bounding box
# ---------------------
def crop_local_patch(images: torch.Tensor, masks_hole: torch.Tensor, patch_size: int = LOCAL_PATCH_SIZE) -> torch.Tensor:
    """
    images: tensor of shape [B, C, H, W],
    masks_hole: tensor of shape [B, 1, H, W],
    returns: tensor of shape [B, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    patches = []
    for b in range(B):
        mask_b = masks_hole[b, 0]
        ys, xs = mask_b.nonzero(as_tuple=True)
        if len(xs) == 0 or len(ys) == 0:
            center_x, center_y = W // 2, H // 2
        else:
            min_x, min_y = xs.min().item(), ys.min().item()
            max_x, max_y = xs.max().item(), ys.max().item()
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

        left = int(max(0, center_x - patch_size // 2))
        top = int(max(0, center_y - patch_size // 2))
        right = left + patch_size
        bottom = top + patch_size

        if right > W:
            right = W
            left = max(0, W-patch_size)
        if bottom > H:
            bottom = H
            top = max(0, H-patch_size)

        patch = images[b:b+1, :, top:bottom, left:right]
        if patch.shape[2] != patch_size or patch.shape[3] != patch_size:
            pad_h = patch_size - patch.shape[2]
            pad_w = patch_size - patch.shape[3]
            # pad format (left, right, top, bottom)
            pad = (0, pad_w, 0, pad_h)
            patch = F.pad(patch, pad, mode='reflect')
        patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches

# ---------------------
# Training function
# ---------------------
def main():
    freeze_support() # for Windows multiprocessing

    # Todo: Setup out folder and logging
    #clear_folder(OUT_PATH)
    # os.makedirs(OUT_PATH, exists_ok=True)

    # Setup seed for randomness (if not pre-defined)
    print(f"PyTorch version: {torch.__version__}")
    seed_val = np.random.randint(1, 10000) if SEED is None else SEED
    print(f"Random Seed: {seed_val}")
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # Setup device (as CUDA if available)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    cuda_available = CUDA and torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.manual_seed(seed_val)
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if cuda_available else "CPU only\n")

    # Setup cudnn benchmark
    cudnn.benchmark = True  # Enable benchmark mode for fixed input size
    # Tells cuDNN to choose the best set of algorithms for the model
    # if the size of input data is fixed. Otherwise, cuDNN will have to find
    # the best algorithms at each iteration.
    # It may dramatically increase the GPU memory consumption, especially when
    # the model architectures are changed during training and
    # both training and evaluation are done in the code.
    # Change it to False if encountered with strange OOM (Out-of-Memory) issues and
    # for varying input sizes.

    # Create a Generator network object
    netG = Generator().to(device)
    netG.apply(weights_init_normal)
    # print(netG) # causes repetitive printing

    # Create Discriminator network objects
    globalD = GlobalDiscriminator().to(device)
    localD = LocalDiscriminator().to(device)
    globalD.apply(weights_init_normal)
    localD.apply(weights_init_normal)
    # print(globalD) # causes repetitive printing

    # Define loss function and optimizer
    criterion = {
        "bce": nn.BCELoss(), # Binary Cross-Entropy loss function
        "l1": nn.L1Loss(), # L1 loss function
        "mse": nn.MSELoss() # MSE loss function
    }

    optimizer_netG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_globalD = optim.Adam(globalD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_localD = optim.Adam(localD.parameters(), lr=lr, betas=(0.5, 0.999))

    # VGG style extractor (frozen)
    style_extractor = VGG19StyleExtractor(device=device).to(device)

    # Dataset and loader
    # CroppedImageDataset already applies transforms (ToTensor + Normalize to [-1, 1])
    dataset = CroppedImageDataset(crops_dir=CROP_PATH)

    # TODO (Another data): Change accordingly to use another data (colored image data)
    # dataset = dset.ImageFolder(root=DATA_PATH,
    #                            transform=transforms.Compose([
    #                                transforms.Resize(X_DIM),
    #                                transforms.CenterCrop(X_DIM),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                            ]))

    assert len(dataset) > 0, f"No crops found in {CROP_PATH} - run preextract-... first!"
    # Try high-performance dataloader
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2, # TODO: Use NUM_WORKERS here after trying!,
            pin_memory=cuda_available
        )
        _ = next(iter(dataloader))  # force load to test
    except Exception as e:
        print("High-performance dataloader error, falling back to num_workers=0:", e)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
    # Add a pin_memory=True argument when calling torch.utils.data.DataLoader()
    # on small datasets, to make sure data is stored at fixed GPU memory addresses
    # and thus increase the data loading speed during training.

    # Fixed samples for visualization
    fixed_masked = []
    fixed_original = []
    fixed_mask_hole = []
    for i in range(min(BATCH_SIZE, len(dataset))):
        m_img, o_img, m_known = dataset[i]
        fixed_masked.append(m_img.unsqueeze(0))
        fixed_original.append(o_img.unsqueeze(0))
        fixed_mask_hole.append((1.0 - m_known).unsqueeze(0))
    fixed_masked = torch.cat(fixed_masked, dim=0).to(device)
    fixed_original = torch.cat(fixed_original, dim=0).to(device)
    fixed_mask_hole = torch.cat(fixed_mask_hole, dim=0).to(device)

    # Trackers for loss values
    losses_log = {
        "G": [],
        "totD": [],
        "globD": [],
        "locD": []
    }

    # TODO: check if necessary to add Fixed noise vector for visualization
    #viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    # Training loop
    print("Starting training...")
    global_step = 0
    start_time = datetime.now()

    for epoch in range(EPOCH_NUM):
        dataloader_tqdm = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1} / {EPOCH_NUM}",
            leave=True,
            ncols=100
        )
        for i, (masked, original, mask_known) in dataloader_tqdm:
            # dataset returns batches of: masked, original, mask_known (1=known)
            masked = masked.to(device)
            original = original.to(device)
            mask_known = mask_known.to(device)

            # convert to hole-mask (1=hole, 0=known) for generator
            mask_hole = (1.0 - mask_known).float().to(device)

            B = original.size(0)
            # Create real and fake label tensors in real time,
            # because there is no guarantee that all sample batches will have the same size
            # BCELoss (Binary Cross Entropy) operates on probabilities,
            # it expects both inputs and targets to be floating-point numbers in the range [0.0, 1.0].
            real_labels = torch.full((B,), REAL_LABEL, device=device, dtype=torch.float)
            fake_labels = torch.full((B,), FAKE_LABEL, device=device, dtype=torch.float)

            # Step 1: Train discriminators
            globalD.zero_grad()
            localD.zero_grad()

            # Generate the full fake image (coarse + fine)
            fake = netG(original, mask_hole)
            # composite: keep known regions from original, fill holes with fake
            composite = fake * mask_hole + original * (1.0 - mask_hole)

            # Create a dictionary to keep losses
            losses = {}

            # Update globalD with real and fake data
            losses["real_globalD"] = criterion["bce"](globalD(original), real_labels)
            losses["fake_globalD"] = criterion["bce"](globalD(composite.detach()), fake_labels)
            losses["globalD"] = 0.5 * (losses["fake_globalD"] + losses["real_globalD"])

            # Update localD with real and fake data
            real_patches = crop_local_patch(original, mask_hole)
            fake_patches = crop_local_patch(composite.detach(), mask_hole)
            losses["real_localD"] = criterion["bce"](localD(real_patches), real_labels)
            losses["fake_localD"] = criterion["bce"](localD(fake_patches), fake_labels)
            losses["localD"] = 0.5 * (losses["fake_localD"] + losses["real_localD"])

            # Loss backwards and optimizer step
            losses["totalD"] = losses["globalD"] + losses["localD"]
            losses["totalD"].backward()
            optimizer_globalD.step()
            optimizer_localD.step()

            # Step 2: Train generator
            netG.zero_grad()

            # Generate the full fake image (coarse + fine)
            fake = netG(original, mask_hole)
            # composite: keep known regions from original, fill holes with fake
            composite = fake * mask_hole + original * (1.0 - mask_hole)

            # Calculate losses
            losses["adv_global"] = criterion["bce"](globalD(composite), real_labels)
            patches = crop_local_patch(composite, mask_hole)
            losses["adv_localD"] = criterion["bce"](localD(patches), real_labels)
            losses["adv"] = losses["adv_globalD"] + losses["adv_localD"]
            losses["l1"] = criterion["l1"](fake * mask_hole, original * mask_hole) * L1_LAMBDA

            # Style loss calculation
            real_features = style_extractor(original)
            fake_features = style_extractor(composite)
            losses["style"] = style_loss(real_features, fake_features, criterion)

            # Loss backwards and optimizer step
            losses["totalG"] = losses["adv"] + losses["l1"] + losses["style"]
            losses["totalG"].backward()
            optimizer_netG.step()

            # Log losses
            losses_log["G"].append(losses["totalG"].item())
            losses_log["totD"].append(losses["totalD"].item())
            losses_log["globD"].append(losses["globalD"].item())
            losses_log["locD"].append(losses["localD"].item())

            # Print the progress
            if i % 100 == 0:
                dataloader_tqdm.set_postfix({
                    'G': losses["totalG"].item(),
                    'totD': losses["totalD"].item(),
                    'globD': losses["globD"].item(),
                    'locD': losses["localD"].item()
                })

                # TODO: Add loss values to graph

                #vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
                #with torch.no_grad():
                #    viz_sample = net_g(viz_noise)
                #    vutils.save_image(viz_sample, os.path.join(OUT_PATH, f'fake_samples_epoch{epoch}_batch{i}.png'),
                #                      normalize=True)
            global_step += 1

        # Save Generator and Discriminator
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(netG.state_dict(), os.path.join(OUT_PATH, f'netG_epoch{epoch+1}.pth'))
            torch.save(globalD.state_dict(), os.path.join(OUT_PATH, f'globalD_epoch{epoch+1}.pth'))
            torch.save(localD.state_dict(), os.path.join(OUT_PATH, f'localD_epoch{epoch+1}.pth'))

    print('Training complete!')

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses_log["G"], label="G")
    plt.plot(losses_log["totD"], label="totD")
    plt.plot(losses_log["globD"], label="globD")
    plt.plot(losses_log["locD"], label="locD")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_PATH, "loss_curve.png"))
    plt.close()

# Train Generator and Discriminator networks
if __name__ == "__main__":
    # Redirect stdout to log file (keep inside __main__ to avoid multiprocessing errors)
    #print(f"Logging to {LOG_FILE}\n")
    #sys.stdout = utils.StdOut(LOG_FILE)
    main()