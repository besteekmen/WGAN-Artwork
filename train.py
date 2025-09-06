# import random (to not use np.random instead)
import os
import sys
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
from os import environ
from datetime import datetime
from tqdm import tqdm
from multiprocessing import freeze_support
#from torch.cuda.amp import autocast, GradScaler
from torch import amp

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
CHECKPOINT_EVERY = globals().get('CHECKPOINT_EVERY', 1) # epochs

# ---------------------
# Utilities:
# ---------------------
class VGGFeatureExtractor(nn.Module):
    """VGG feature extractor to get feature maps

    mode: string, type of feature extraction, style or perceptual
    layers: controls which layers are extracted
    returns: feature maps at chosen layers
    """
    def __init__(self, mode="style", layers=None, pretrained=True):
        super().__init__()
        # Set the feature type related variables
        if mode == "style":
            vgg = tvmodels.vgg19(pretrained=pretrained).features.eval()
            # common relu layers to use for style Gram: relu1_1, relu2_1, relu3_1, relu4_1
            # module indices: relu1_1='1', relu2_1='6', relu3_1='11', relu4_1='20'
            layers = layers or [1, 6, 11, 20]
        elif mode == "perceptual":
            vgg = tvmodels.vgg16(pretrained=pretrained).features.eval()
            # Default layers for perceptual loss (relu1_2, relu2_2, relu3_3, relu4_3)
            layers = layers or [3, 8, 15, 22]
        else:
            raise ValueError(f"Mode must be 'style' or 'perceptual', unsupported: {mode}")

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.layers = [int(x) for x in layers]
        # store ImageNet mean/std statistics for normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Rescale from [-1, 1] to [0, 1]
        x = x.float()
        x = (x + 1.0) / 2.0
        # Normalize with ImageNet statistics
        x = (x - self.mean) / self.std
        # Pass through VGG and extract chosen layers
        features = []
        cur = x
        for idx, layer in enumerate(self.vgg):
            cur = layer(cur)
            if idx in self.layers:
                features.append(cur)
        return features

def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Calculate Gram matrix for style loss.
    Input (features): tensor of shape [B, C, H, W]
    Output: tensor of shape [B, C, C] TODO: check!"""
    B, C, H, W = features.size()
    feats = features.view(B, C, H * W)
    # add a small eps,lon to avoid NaN in float16
    return torch.bmm(feats, feats.transpose(1, 2)) / (C * H * W + 1e-4)
    # TODO: should divide by H*W?

def feature_loss(real_features, fake_features, mode, criterion):
    """Compute feature loss between real and fake feature maps."""
    if mode == "style":
        compare = lambda x, y: criterion(gram_matrix(x), gram_matrix(y))
    elif mode == "perceptual":
        compare = lambda x, y: criterion(x, y)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    loss = 0
    for rf, ff in zip(real_features, fake_features):
        loss += compare(rf, ff)
    return loss / len(real_features) # normalize by number of layers v6

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
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(B, -1)
    grad_norm = gradients.norm(2, dim=1)
    grad_norm = torch.clamp(grad_norm, min=1e-8) # To avoid NaN without bias
    #grad_norm = gradients.norm(2, dim=1) + 1e-8 # stabilizer added to avoid nan g loss TODO:check!
    return ((grad_norm - 1) ** 2).mean()

# ---------------------
# Helper functions:
# ---------------------
def get_weights(schedule, epoch):
    """Using the schedule, returns the related weights at current epoch."""
    for e, w in reversed(schedule):
        if epoch >= e: # check if current epoch has reached this schedule
            return w
    return schedule[0][1] # return the first weight

def erode_mask(mask: torch.Tensor, k: int=3) -> torch.Tensor:
    """Morphological erosion (1 = hole, 0 = known) by k x k kernels.
    Shrinks the hole region, leaving a safe margin around boundaries."""

    # max_pool2d on inverted mask = erosion
    return -F.max_pool2d(mask, kernel_size=k, stride=1, padding=k//2)

def crop_local_patch2(images: torch.Tensor, masks_hole: torch.Tensor, patch_size: int = LOCAL_PATCH_SIZE) -> torch.Tensor:
    """
    Local patch extraction for a batch of images.

    images: tensor of shape [B, C, H, W],
    masks_hole: tensor of shape [B, 1, H, W], 0 = hole, 1 = valid
    patch_size: int, size of local patch
    returns: tensor of shape [B, C, patch_size, patch_size]
    """
    B, C, H, W = images.shape
    masks_hole = masks_hole.squeeze(1) # [B, H, W]

    # Find the mask location using row/col sums
    rows = (masks_hole == 0).any(dim=2) # [B, H]
    cols = (masks_hole == 0).any(dim=1) # [B, W]

    # First and last indices where hole appears
    top = torch.argmax(rows.float(), dim=1)
    bottom = H - torch.argmax(torch.flip(rows.float(), [1]), dim=1) - 1
    left = torch.argmax(cols.float(), dim=1)
    right = W - torch.argmax(torch.flip(cols.float(), [1]), dim=1) - 1

    # Center coordinates
    cy = ((top + bottom) // 2).clamp(0, H-1)
    cx = ((left + right) // 2).clamp(0, W-1)

    # Top-left corners for patches
    y0 = (cy - patch_size // 2).clamp(0, H-patch_size)
    x0 = (cx - patch_size // 2).clamp(0, W-patch_size)

    patches = []
    for i in range(B):
        y_start = y0[i].item()
        x_start = x0[i].item()
        patch = images[i:i+1, :, y_start:y_start+patch_size, x_start:x_start+patch_size]
        patches.append(patch)
    return torch.cat(patches, dim=0) # [B, C, patch_size, patch_size]

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

    # ---------------------
    # Setup
    # ---------------------
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
    style_extractor = VGGFeatureExtractor(mode="style").to(device)
    # VGG perceptual extractor (frozen)
    perceptual_extractor = VGGFeatureExtractor(mode="perceptual").to(device)

    # TODO: Should I explicitly call on eval? (does it drop extra layers?)
    #style_extractor.eval()
    #perceptual_extractor.eval()

    # Grad scaler defined for faster training and lower memory usage
    scalerG = amp.GradScaler()
    scaler_globalD = amp.GradScaler()
    scaler_localD = amp.GradScaler()
    # They will scale losses for mixed precision

    # Datasets and loaders
    # CroppedImageDataset already applies transforms (ToTensor + Normalize to [-1, 1])
    # To use a different transform, create one here and add as parameter below!
    train_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'train'), split='train')
    val_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'val'), split='val')

    train_loader = make_dataloader(train_set, 'train', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available)
    val_loader = make_dataloader(val_set, 'val', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available, shuffle=False)

    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
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

    # ---------------------
    # Training loop
    # ---------------------
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

        hole_lambda = get_weights(LOSS_SCHEDULE["HOLE_LAMBDA"], epoch)
        valid_lambda = get_weights(LOSS_SCHEDULE["VALID_LAMBDA"], epoch)
        l1_lambda = get_weights(LOSS_SCHEDULE["L1_LAMBDA"], epoch)
        global_lambda = get_weights(LOSS_SCHEDULE["GLOBAL_LAMBDA"], epoch)
        local_lambda = get_weights(LOSS_SCHEDULE["LOCAL_LAMBDA"], epoch)
        style_lambda = get_weights(LOSS_SCHEDULE["STYLE_LAMBDA"], epoch)
        adv_lambda = get_weights(LOSS_SCHEDULE["ADV_LAMBDA"], epoch)
        perceptual_lambda = get_weights(LOSS_SCHEDULE["PERCEPTUAL_LAMBDA"], epoch)
        inter_weight = get_weights(SCALE_SCHEDULE["inter"], epoch)
        fine_weight = get_weights(SCALE_SCHEDULE["fine"], epoch)
        gp_freq = get_weights(GP_SCHEDULE, epoch)

        for i, (masked, original, mask_known) in train_tqdm:
            # dataset returns batches of: masked, original, mask_known (1=known)
            masked = masked.to(device)
            original = original.to(device)
            mask_known = mask_known.to(device)
            # convert to hole-mask (1=hole, 0=known) for generator
            mask_hole = (1.0 - mask_known).float().to(device)

            # Forward generator once per batch
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                fake = netG(original, mask_hole)
                #fake, intermediate = netG(original, mask_hole)  # fine 256x256 and intermediate 128x128
                # composite: keep known regions from original, fill holes with fake
                composite = fake * mask_hole + original * (1.0 - mask_hole)

            # Create detached versions to avoid multiple forwarding of generator
            composite_detached = composite.detach()

            # Create a dictionary to keep losses
            losses = {}

            # -------------------------------------------------
            # Step 1: Train discriminators as critics (WGAN-GP)
            # -------------------------------------------------
            optimizer_globalD.zero_grad()
            optimizer_localD.zero_grad()

            # Global D
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                # Update globalD for global critic
                # backward and optimizer steps are called separately to not mix!
                real_global = globalD(original)
                fake_global = globalD(composite_detached)
                if global_step % gp_freq == 0:
                    idx = torch.randperm(original.size(0))[:GP_SUBSET] # random subset
                    real_subset = original[idx]
                    fake_subset = composite_detached[idx]
                    gp_global = gradient_penalty(globalD, real_subset, fake_subset, device)
                else:
                    gp_global = 0.0

                loss_globalD = (fake_global.mean() - real_global.mean()) + GP_LAMBDA * gp_global

            # DO NOT detach before backward to make sure gradients flow!
            scaler_globalD.scale(loss_globalD).backward()
            scaler_globalD.step(optimizer_globalD)
            scaler_globalD.update()
            losses["globalD"] = loss_globalD.detach()

            # Update localD for local critic
            # Using detached versions for discriminator is okay, but not okay for generator
            real_patches = crop_local_patch(original, mask_hole)
            fake_patches = crop_local_patch(composite_detached, mask_hole)
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                real_local = localD(real_patches)
                fake_local = localD(fake_patches)
                if global_step % gp_freq == 0:
                    idx = torch.randperm(real_patches.size(0))[:GP_SUBSET] # random subset
                    real_subset = real_patches[idx]
                    fake_subset = fake_patches[idx]
                    gp_local = gradient_penalty(localD, real_subset, fake_subset, device)
                else:
                    gp_local = 0.0

                loss_localD = (fake_local.mean() - real_local.mean()) + GP_LAMBDA * gp_local

            # DO NOT detach before backward to make sure gradients flow!
            scaler_localD.scale(loss_localD).backward()
            scaler_localD.step(optimizer_localD)
            scaler_localD.update()
            losses["localD"] = loss_localD.detach()

            # Final discriminator loss
            losses["totalD"] = (losses["globalD"] + losses["localD"]).detach()

            # Free memory for discriminator step
            del composite_detached, real_global, fake_global, gp_global
            del real_patches, fake_patches, real_local, fake_local, gp_local
            torch.cuda.empty_cache()

            # -------------------------------------------------
            # Step 2: Train generator
            # -------------------------------------------------
            optimizer_netG.zero_grad()

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                # Use already existing fake and composite!
                # Downsample original and mask for intermediate scale
                original_inter = F.interpolate(original, size=(128, 128), mode="bilinear", align_corners=False)
                mask_inter = F.interpolate(mask_known, size=(128, 128), mode="nearest")
                fake_inter = F.interpolate(fake, size=(128, 128), mode="bilinear", align_corners=False)
                # TODO: to switch back to multi-resolution return from generator, get fake_inter from netG

                # Erode masks to ignore 1px halo at boundaries
                eroded_mask = erode_mask(mask_hole, k=3)
                eroded_inter = erode_mask(mask_inter, k=3)

                # -------------------------------------------------
                # Adversarial Loss (negated critic scores)
                # -------------------------------------------------
                adv_global = -globalD(composite).mean()
                adv_local = -localD(crop_local_patch(composite, mask_hole)).mean()
                losses["adv"] = global_lambda * adv_global + local_lambda * adv_local

                # -------------------------------------------------
                # Pixel-wise L1 loss
                # -------------------------------------------------
                # Weighted l1 loss: https://arxiv.org/pdf/2401.03395
                # Also weighted: https://arxiv.org/pdf/1801.07892
                # Training only the masked area may cause seam artifacts at boundary and inconsistencies
                # Fine scale: (256 x 256) on full image
                l1_fine = hole_lambda * criterion["l1"](fake * eroded_mask, original * eroded_mask) + \
                          valid_lambda * criterion["l1"](fake * (1.0 - eroded_mask), original * (1.0 - eroded_mask))
                # Coarse scale: (128, 128) only L1
                l1_inter = hole_lambda * criterion["l1"](fake_inter * eroded_inter, original_inter * eroded_inter) + \
                          valid_lambda * criterion["l1"](fake_inter * (1.0 - eroded_inter), original_inter * (1.0 - eroded_inter))
                losses["l1"] = inter_weight * l1_inter.float() + \
                               fine_weight * l1_fine.float()

            # -------------------------------------------------
            # Style loss (no autocast to avoid NaN, only fine)
            # -------------------------------------------------
            #with torch.no_grad(): # as VGG is frozen, no need to track gradients here
            #real_style = style_extractor(original).detach()
            real_style = [f.detach() for f in style_extractor(original)]
            fake_style = style_extractor(composite)
                # detach composite to break any graph if you remove no_grad!
            sl = feature_loss(real_style, fake_style, "style", criterion["mse"])
            losses["style"] = sl  # used only on final scale

            # -------------------------------------------------
            # Perceptual loss (no autocast, only fine)
            # -------------------------------------------------
            # Perceptual loss helps generator to match semantic structures instead of only textures or pixels.
            with torch.no_grad():  # as VGG is frozen, no need to track gradients here
                real_perc = perceptual_extractor(original)
            fake_perc = perceptual_extractor(composite)
            # DO NOT detach composite here! Gradients must flow to generator!
            pl = feature_loss(real_perc, fake_perc, "perceptual", criterion["mse"])
            losses["perceptual"] = pl

            # -------------------------------------------------
            # Aggregate generator losses
            # -------------------------------------------------
            losses["totalG"] = (adv_lambda * losses["adv"].float() +
                                l1_lambda * losses["l1"] +
                                style_lambda * losses["style"].float() +
                                perceptual_lambda * losses["perceptual"].float())

            # Loss backwards and optimizer step
            scalerG.scale(losses["totalG"]).backward()
            scalerG.step(optimizer_netG)
            scalerG.update()

            # Free memory for generator step
            del adv_local, adv_global, real_style, fake_style, real_perc, fake_perc
            del l1_fine, l1_inter, pl, sl
            #torch.cuda.empty_cache()

            # Log losses
            losses_log["totalG"].append(losses["totalG"].item())
            losses_log["totalD"].append(losses["totalD"].item())
            losses_log["globalD"].append(losses["globalD"].item())
            losses_log["localD"].append(losses["localD"].item())

            # Print the progress
            if i % SAVE_FREQ == 0:
                train_tqdm.set_postfix({
                    'G': losses["totalG"].item(),
                    'D': losses["totalD"].item(),
                    'gD': losses["globalD"].item(),
                    'lD': losses["localD"].item()
                })

                # save a batch of image comparisons
                with torch.no_grad():
                    vis_comp = composite # clone can be used if the tensor is manipulated later
                    # concatenate images horizontally per sample: cat along width axis (dim=3)
                    grid = torch.cat(
                        [to_unit(original), to_unit(masked), to_unit(vis_comp)],
                        3) # horizontal stack per sample using width dimension
                    vutils.save_image(grid,
                                      os.path.join(OUT_PATH, f'comparison_epoch({epoch})_batch({i}).png'),
                                      normalize=True,
                                      nrow=1) # 1 row per sample (not BATCH_SIZE)?
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

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------
        val_losses = []
        with (torch.no_grad()):
            for j, (masked, original, mask_known) in enumerate(val_loader):
                masked = masked.to(device)
                original = original.to(device)
                mask_known = mask_known.to(device)
                mask_hole = (1.0 - mask_known).float().to(device)

                # Generator forward w/o grad
                fake = netG(original, mask_hole)
                #fake, intermediate = netG(original, mask_hole)  # fine 256x256 and intermediate 128x128
                composite = fake * mask_hole + original * (1.0 - mask_hole)

                # Downsample original and mask
                original_inter = F.interpolate(original, size=(128, 128), mode="bilinear", align_corners=False)
                mask_inter = F.interpolate(mask_hole, size=(128, 128), mode="nearest")
                fake_inter = F.interpolate(fake, size=(128, 128), mode="bilinear", align_corners=False)

                # Erode masks to ignore 1px halo at boundaries
                eroded_mask = erode_mask(mask_hole, k=3)
                eroded_inter = erode_mask(mask_inter, k=3)

                # -------------------------------------------------
                # Pixel-wise L1 loss
                # -------------------------------------------------
                # Weighted l1 loss: https://arxiv.org/pdf/2401.03395
                # Also weighted: https://arxiv.org/pdf/1801.07892
                # Training only the masked area may cause seam artifacts at boundary and inconsistencies
                # L1 loss on final output (256 x 256)
                vl1_fine = hole_lambda * criterion["l1"](fake * eroded_mask, original * eroded_mask) + \
                           valid_lambda * criterion["l1"](fake * (1.0 - eroded_mask),
                                                          original * (1.0 - eroded_mask))
                # L1 loss on intermediate output (128, 128)
                vl1_inter = hole_lambda * criterion["l1"](fake_inter * eroded_inter, original_inter * eroded_inter) + \
                            valid_lambda * criterion["l1"](fake_inter * (1.0 - eroded_inter),
                                                           original_inter * (1.0 - eroded_inter))
                vl1_loss = inter_weight * vl1_inter.float() + \
                           fine_weight * vl1_fine.float()

                # -------------------------------------------------
                # Style loss
                # -------------------------------------------------
                real_style = style_extractor(original)
                fake_style = style_extractor(composite)
                vsl = feature_loss(real_style, fake_style, "style", criterion["mse"])

                # -------------------------------------------------
                # Perceptual loss
                # -------------------------------------------------
                real_perc = perceptual_extractor(original)
                fake_perc = perceptual_extractor(composite)
                vpl = feature_loss(real_perc, fake_perc, "perceptual", criterion["mse"])

                # -------------------------------------------------
                # Aggregate losses
                # -------------------------------------------------
                totalG_val_loss = (l1_lambda * vl1_loss +
                                   style_lambda * vsl.float() +
                                   perceptual_lambda * vpl.float())
                val_losses.append(totalG_val_loss.item())

                # Free memory for validation step
                del fake, composite, real_style, fake_style, real_perc, fake_perc
                del vpl, vsl, vl1_loss, vl1_inter, vl1_fine
                #torch.cuda.empty_cache()

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
            fixed_fake = netG(fixed_original, fixed_mask_hole)
            #fixed_fake, fixed_inter = netG(fixed_original, fixed_mask_hole)  # fine 256x256 and intermediate 128x128
            fixed_comp = (fixed_fake * fixed_mask_hole + fixed_original * (1.0 - fixed_mask_hole)).detach()

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
    # Redirect stdout to log file (keep inside __main__ to avoid multiproc+essing errors)
    #print(f"Logging to {LOG_FILE}\n")
    #sys.stdout = utils.StdOut(LOG_FILE)
    main()