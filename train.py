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
from torch.cuda.amp import autocast, GradScaler

# Import metrics to evaluate
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from config import *
from utils import clear_folder, to_unit, to_signed
from data_utils import CroppedImageDataset, make_dataloader
from models.generator import Generator
from models.discriminator import GlobalDiscriminator, LocalDiscriminator
from models.weights_init import weights_init_normal
# TODO: Should I use this below:
#os.environ["PYTHONUNBUFFERED"] = "1"

# Optional config fallbacks
SAMPLE_SAVE_STEP = globals().get('SAMPLE_SAVE_STEP', 200) # batches
CHECKPOINT_EVERY = globals().get('CHECKPOINT_EVERY', 1) # epochs

# ---------------------
# Utilities: Gradient penalty, gram matrix and VGG extractors
# ---------------------
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
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Calculate Gram matrix for style loss.
    Input (features): tensor of shape [B, C, H, W]
    Output: tensor of shape [B, C, C] TODO: check!"""
    B, C, H, W = features.size()
    feats = features.view(B, C, H * W)
    return torch.bmm(feats, feats.transpose(1, 2)) / (C * H * W)
    # TODO: should divide by H*W?

class VGG19StyleExtractor(nn.Module):
    """Extract intermediate feature maps from VGG19 for style loss (frozen)."""
    def __init__(self, layers=None):
        super().__init__()
        vgg = tvmodels.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        # common relu layers to use for style Gram: relu1_1, relu2_1, relu3_1, relu4_1
        # module indices: relu1_1='1', relu2_1='6', relu3_1='11', relu4_1='20'
        self.layers = [int(x) for x in (layers or [1, 6, 11, 20])]
        #layers or ['1', '6', '11', '20']
        # store ImageNet mean/std statistics for normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Rescale from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Normalize with ImageNet statistics
        x = (x - self.mean) / self.std
        # Pass through VGG and extract chosen layers
        features = []
        cur = x
        #for name, layer in self.vgg.named_children():
        for idx, layer in enumerate(self.vgg):
            cur = layer(cur)
            #if name in self.layers:
            if idx in self.layers:
                features.append(cur)
                #if len(features) == len(self.layers):
                #    break
        return features

# Perceptual loss helps generator to match semantic structures instead of only textures or pixels.
# Yu et al 2018 Contextual attention, Liu et al 2018 partial convolutions etc uses both style and perceptual
class VGG16PerceptualLoss(nn.Module):
    """Perceptual loss for VGG16.
    Compare feature maps of real and generated images."""
    def __init__(self, layers=None, resize=True):
        super().__init__()
        vgg = tvmodels.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        # Default layers for perceptual loss (relu1_2, relu2_2, relu3_3, relu4_3)
        self.layers = [int(x) for x in (layers or [3, 8, 15, 22])]
        # store ImageNet mean/std statistics for normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.MSELoss() # TODO: use criterion dict later?

    def forward(self, real, fake):
        """
        real, fake: [B, 3, H, W] values in [-1, 1]
        """
        # Rescale from [-1, 1] to [0, 1] and normalize
        real = (real + 1.0) / 2.0
        fake = (fake + 1.0) / 2.0
        # Normalize with ImageNet statistics
        real = (real - self.mean) / self.std
        fake = (fake - self.mean) / self.std

        # Pass through VGG and extract features
        real_features = []
        fake_features = []
        x_real = real
        x_fake = fake
        for idx, layer in enumerate(self.vgg):
            x_real = layer(x_real)
            x_fake = layer(x_fake)
            if idx in self.layers:
                real_features.append(x_real)
                fake_features.append(x_fake)

        # Compute MSE between features
        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += self.criterion(rf, ff)
        return loss

# ---------------------
# Helper functions:
# ---------------------
def style_loss(real_features, fake_features, criterion):
    """Compute style loss between real and fake feature maps."""
    loss = 0
    for rf, ff in zip(real_features, fake_features):
        loss += criterion["mse"](gram_matrix(rf), gram_matrix(ff))
    return loss

def style_perceptual_loss(original, composite, style_extractor, perceptual_loss_func, criterion):
    """Computes style and perceptual losses per batch.
    original, composite: [B, 3, H, W] values in [-1, 1]"""

    # Compute VGG features in mixed precision for less GPU load
    with autocast():
        # Style loss
        real_features = style_extractor(original)
        fake_features = style_extractor(composite)
        style = style_loss(real_features, fake_features, criterion)

        # Perceptual loss
        perceptual = perceptual_loss_func(original, composite)

    return style, perceptual

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

    # Todo: Setup logging
    # Make sure output folder exists and clean
    clear_folder(OUT_PATH)
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

    # enable below if exact reproducibility is needed, also disable benchmark
    #cudnn.deterministic = True
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
        # WGAN-GP: Do not use BCE to switch to the WGAN-GP principles!
        # Hence BCE is removed. Also rather than BCEWithLogitsLoss, WGAN used
        # TODO: Should I use BCEWithLogitsLoss and remove sigmoid layer?
        "l1": nn.L1Loss(), # L1 loss function
        "mse": nn.MSELoss() # MSE loss function
    }

    # Optimizers
    # If discriminator gets perfect quickly, generator gradients may vanish!
    # Hence make learning slower for D and faster for G (Contextual Attention GAN, Yu et al. 2018)
    # TODO: If good texture but broken structure, prioritize global (Izuka et al. 2017)
    # TODO: If discriminators are weak (not decreasing loss), use update ratios (WGAN-GP)
    # WGAN-GP: betas changed to (0.0, 0.9) from (0.5, 0.999)
    optimizer_netG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.0, 0.9))
    optimizer_globalD = optim.Adam(globalD.parameters(), lr=LR_D, betas=(0.0, 0.9))
    optimizer_localD = optim.Adam(localD.parameters(), lr=LR_D, betas=(0.0, 0.9))

    # VGG style extractor (frozen)
    style_extractor = VGG19StyleExtractor().to(device)
    # VGG perceptual loss
    perceptual_loss_func = VGG16PerceptualLoss().to(device)

    # Grad scaler defined for faster training and lower memory usage
    scalerG = GradScaler()
    scaler_globalD = GradScaler()
    scaler_localD = GradScaler()
    # They will scale losses for mixed precision

    # Datasets and loaders
    # CroppedImageDataset already applies transforms (ToTensor + Normalize to [-1, 1])
    # To use a different transform, create one here and add as parameter below!
    train_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'train'), split='train')
    val_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'val'), split='val')

    train_loader = make_dataloader(train_set, 'train', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available)
    val_loader = make_dataloader(val_set, 'val', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available)

    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Fixed samples for visualization
    fixed_masked = []
    fixed_original = []
    fixed_mask_hole = []
    for i in range(min(BATCH_SIZE, len(train_set))):
        m_img, o_img, m_known = train_set[i]
        fixed_masked.append(m_img.unsqueeze(0))
        fixed_original.append(o_img.unsqueeze(0))
        fixed_mask_hole.append((1.0 - m_known).unsqueeze(0))
    fixed_masked = torch.cat(fixed_masked, dim=0).to(device)
    fixed_original = torch.cat(fixed_original, dim=0).to(device)
    fixed_mask_hole = torch.cat(fixed_mask_hole, dim=0).to(device)

    # Trackers for loss values
    losses_log = {
        "totalG": [],
        "totalD": [],
        "globalD": [],
        "localD": []
    }

    # Trackers for averaged loss values per epoch
    epoch_log = {
        "totalG": [],
        "totalD": [],
        "globalD": [],
        "localD": [],
        "valG": [] # added for validation
    }

    # Training loop
    print("Starting training...")
    global_step = 0
    start_time = datetime.now()

    for epoch in range(EPOCH_NUM):
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1} / {EPOCH_NUM}",
            leave=True,
            ncols=100
        )
        for i, (masked, original, mask_known) in train_tqdm:
            # dataset returns batches of: masked, original, mask_known (1=known)
            masked = masked.to(device)
            original = original.to(device)
            mask_known = mask_known.to(device)
            # convert to hole-mask (1=hole, 0=known) for generator
            mask_hole = (1.0 - mask_known).float().to(device)

            # Create a dictionary to keep losses
            losses = {}

            # Step 1: Train discriminators as critics (WGAN-GP)
            globalD.zero_grad()
            localD.zero_grad()

            # Scaler: with autocast(): (add for the following 4 lines up to gap)
            # Generate the full fake image (coarse + fine)
            fake = netG(original, mask_hole)
            # composite: keep known regions from original, fill holes with fake
            composite = fake * mask_hole + original * (1.0 - mask_hole)

            # Create detached versions to avoid multiple forwarding of generator
            fake_detached = fake.detach()
            composite_detached = fake_detached * mask_hole + original * (1.0 - mask_hole)

            # Scaler: with autocast(): (add for the following lines up to gap loss backward)
            # Update globalD for global critic
            # backward and optimizer steps are called separately to not mix!
            real_global = globalD(original)
            fake_global = globalD(composite_detached)
            gp_global = gradient_penalty(globalD, original, composite_detached, device)
            loss_globalD = (fake_global.mean() - real_global.mean()) + GP_LAMBDA * gp_global
            loss_globalD.backward()
            optimizer_globalD.step()
            # Scaler: add below lines instead of above backward and step
            # scaler_globalD.scale(loss_globalD).backward()
            # scaler_globalD.step(optimizer_globalD)
            # scaler_globalD.update()
            losses["globalD"] = loss_globalD.detach()

            # Update localD for local critic
            # Using detached versions for discriminator is okay, but not okay for generator
            real_patches = crop_local_patch(original, mask_hole)
            fake_patches = crop_local_patch(composite_detached, mask_hole)
            # Scaler: with autocast(): (add for the following lines up to gap loss backward)
            real_local = localD(real_patches)
            fake_local = localD(fake_patches)
            gp_local = gradient_penalty(localD, real_patches, fake_patches, device)
            loss_localD = (fake_local.mean() - real_local.mean()) + GP_LAMBDA * gp_local
            loss_localD.backward()
            optimizer_localD.step()
            # Scaler: add below lines instead of above backward and step
            # scaler_localD.scale(loss_localD).backward()
            # scaler_localD.step(optimizer_localD)
            # scaler_localD.update()
            losses["localD"] = loss_localD.detach()

            # Final discriminator loss
            losses["totalD"] = (losses["globalD"] + losses["localD"]).detach()

            # Step 2: Train generator
            netG.zero_grad()
            # Scaler: with autocast(): (add for the following lines up to gap loss backward)

            # Adversarial (negated critic scores)
            adv_global = -globalD(composite).mean()
            patches = crop_local_patch(composite, mask_hole)
            adv_local = -localD(patches).mean()
            losses["adv"] = adv_global + adv_local

            # Pixel-wise reconstruction loss
            # Weighted l1 loss: https://arxiv.org/pdf/2401.03395
            # Also weighted: https://arxiv.org/pdf/1801.07892
            # Training only the masked area may cause seam artifacts at boundary and inconsistencies
            losses["hole"] = criterion["l1"](fake * mask_hole, original * mask_hole)
            losses["valid"] = criterion["l1"](fake * (1.0 - mask_hole), original * (1.0 - mask_hole))
            losses["l1"] = HOLE_LAMBDA * losses["hole"] + VALID_LAMBDA * losses["valid"]

            # Style and perceptual loss per batch
            losses["style"], losses["perceptual"] = style_perceptual_loss(
                original, composite, style_extractor, perceptual_loss_func, criterion
            )

            # Final generator loss
            losses["totalG"] = (losses["adv"] +
                                L1_LAMBDA * losses["l1"] +
                                STYLE_LAMBDA * losses["style"] +
                                PERCEPTUAL_LAMBDA * losses["perceptual"])
            # Loss backwards and optimizer step
            losses["totalG"].backward()
            optimizer_netG.step()
            # Scaler: add below lines instead of above backward and step
            # scaler_G.scale(losses["totalG"]).backward()
            # scaler_G.step(optimizer_netG)
            # scaler_G.update()

            # Log losses
            losses_log["totalG"].append(losses["totalG"].item())
            losses_log["totalD"].append(losses["totalD"].item())
            losses_log["globalD"].append(losses["globalD"].item())
            losses_log["localD"].append(losses["localD"].item())

            # Print the progress
            if i % 100 == 0:
                train_tqdm.set_postfix({
                    'G': losses["totalG"].item(),
                    'D': losses["totalD"].item(),
                    'gD': losses["globalD"].item(),
                    'lD': losses["localD"].item()
                })

                # save a batch of image comparisons
                with torch.no_grad():
                    vis_fake = fake # or fake.detach().clone()?
                    vis_comp = composite # clone can be used if the tensor is manipulated later
                    grid = torch.cat(
                        [to_unit(original), to_unit(masked), to_unit(vis_comp)],
                        1) # horizontal stack per sample
                    vutils.save_image(grid,
                                      os.path.join(OUT_PATH, f'comparison_epoch({epoch})_batch({i}).png'),
                                      normalize=True,
                                      nrow=1) # 1 row per sample (not BATCH_SIZE)
            global_step += 1

        # Compute and store average losses for the current epoch
        avg_totalG = sum(losses_log["totalG"][-len(train_loader):]) / len(train_loader)
        avg_totalD = sum(losses_log["totalD"][-len(train_loader):]) / len(train_loader)
        avg_globalD = sum(losses_log["globalD"][-len(train_loader):]) / len(train_loader)
        avg_localD = sum(losses_log["localD"][-len(train_loader):]) / len(train_loader)

        epoch_log["totalG"].append(avg_totalG)
        epoch_log["totalD"].append(avg_totalD)
        epoch_log["globalD"].append(avg_globalD)
        epoch_log["localD"].append(avg_localD)

        # Validation loop
        val_losses = []
        with torch.no_grad():
            for masked, original, mask_known in val_loader:
                masked = masked.to(device)
                original = original.to(device)
                mask_known = mask_known.to(device)
                mask_hole = (1.0 - mask_known).float().to(device)

                fake = netG(original, mask_hole)
                composite = fake * mask_hole + original * (1.0 - mask_hole)

                # Adversarial (negated critic scores)
                adv_global = -globalD(composite).mean()
                patches = crop_local_patch(composite, mask_hole)
                adv_local = -localD(patches).mean()
                adv_loss = adv_global + adv_local

                hole_loss = criterion["l1"](fake * mask_hole, original * mask_hole)
                valid_loss = criterion["l1"](fake * (1.0 - mask_hole), original * (1.0 - mask_hole))
                l1_loss = HOLE_LAMBDA * hole_loss + VALID_LAMBDA * valid_loss

                # Style and perceptual loss calculation per batch
                style_l, perceptual_l = style_perceptual_loss(
                    original, composite, style_extractor, perceptual_loss_func, criterion
                )

                # Final generator loss
                totalG_val_loss = (adv_loss +
                                   L1_LAMBDA * l1_loss +
                                   STYLE_LAMBDA * style_l +
                                   PERCEPTUAL_LAMBDA * perceptual_l)
                val_losses.append(totalG_val_loss.item())

        avg_valG = sum(val_losses) / len(val_losses)
        epoch_log["valG"].append(avg_valG)
        print(f"Validation Epoch: {epoch+1}: G={avg_valG:.4f}")

        # Save Generator and Discriminator
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(netG.state_dict(), os.path.join(OUT_PATH, f'netG_epoch{epoch+1}.pth'))
            torch.save(globalD.state_dict(), os.path.join(OUT_PATH, f'globalD_epoch{epoch+1}.pth'))
            torch.save(localD.state_dict(), os.path.join(OUT_PATH, f'localD_epoch{epoch+1}.pth'))

        # Save fixed samples to track progress on same inputs
        with torch.no_grad():
            fixed_fake = netG(fixed_original, fixed_mask_hole).detach()
            fixed_comp = fixed_fake * fixed_mask_hole + fixed_original * (1.0 - fixed_mask_hole).detach()

            unit_original = to_unit(fixed_original)
            unit_comp = to_unit(fixed_comp)
            unit_masked = to_unit(fixed_masked)

            ssim_val = ssim(unit_comp, unit_original).item()
            lpips_val = lpips(unit_comp, unit_original).item()

            fid_val = None
            if (epoch + 1) % 5 == 0:
                fid.reset()
                fid.update(unit_original, real=True)
                fid.update(unit_comp, real=False)
                fid_val = fid.compute().item()

            print(f"Epoch {epoch+1}: SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}", end="")
            if fid_val is not None:
                print(f", FID={fid_val:.2f}")
            else:
                print("")

            grid = torch.cat(
                [unit_original, unit_masked, unit_comp],
                0)
            vutils.save_image(
                grid,
                os.path.join(OUT_PATH, f'fixed_epoch{epoch+1}.png'),
                normalize=True,
                nrow=fixed_original.size(0) # TODO: Or BATCH_SIZE?
            )

    print('Training complete!')

    # Plot per-batch loss curve
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (per batch)")
    plt.plot(losses_log["totalG"], label="G (batch)")
    plt.plot(losses_log["totalD"], label="D (batch)")
    plt.plot(losses_log["globalD"], label="gD (batch)")
    plt.plot(losses_log["localD"], label="lD (batch)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_PATH, "loss_curve_batch.png"))
    plt.close()

    # Plot per-epoch averaged loss curve (Like DCGAN)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (per epoch avg)")
    plt.plot(epoch_log["totalG"], label="G (epoch avg)")
    plt.plot(epoch_log["totalD"], label="D (epoch avg)")
    plt.plot(epoch_log["globalD"], label="gD (epoch avg)")
    plt.plot(epoch_log["localD"], label="lD (epoch avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_PATH, "loss_curve_epoch.png"))
    plt.close()

# Train Generator and Discriminator networks
if __name__ == "__main__":
    # Redirect stdout to log file (keep inside __main__ to avoid multiprocessing errors)
    #print(f"Logging to {LOG_FILE}\n")
    #sys.stdout = utils.StdOut(LOG_FILE)
    main()