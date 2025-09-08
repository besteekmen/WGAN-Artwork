import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from datetime import datetime

from tqdm import tqdm
from multiprocessing import freeze_support
from torch import amp

# Import metrics to evaluate
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from config import *
from losses import VGG19StyleLoss, VGG16PerceptualLoss, gradient_penalty
from utils.utils import clear_folder, to_unit, downsample
from utils.data_utils import CroppedImageDataset, make_dataloader
from utils.vision_utils import crop_local_patch
from models.generator import Generator
from models.discriminator import GlobalDiscriminator, LocalDiscriminator
from models.weights_init import weights_init_normal

# Optional config fallbacks
CHECKPOINT_EVERY = globals().get('CHECKPOINT_EVERY', 1) # epochs

# ------------------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------------------
def main():
    freeze_support() # for Windows multiprocessing

    #---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------
    # Todo: Setup logging

    # Make sure output folder exists and clean
    clear_folder(OUT_PATH)

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
    cudnn.benchmark = True
    # cuDNN chooses the best set of algorithms for the model (only fixed input)
    # If input size NOT fixed: will have to find the best at each iteration and \
    # the GPU memory consumption increases, especially when the models are changed \
    # during training, and both training and evaluation done in the code.
    # Set to false if strange OOM (Out of memory) issues seen

    # Setup cudnn deterministic
    # Set to true if exact reproducibility is needed and disable benchmark
    #cudnn.deterministic = True

    # Setup Generator network object
    netG = Generator().to(device)
    netG.apply(weights_init_normal)
    netG.upsample_init() # used to avoid initial patchy results
    # print(globalD) # DEBUG only: causes repetitive printing

    # Setup Discriminator network objects
    globalD = GlobalDiscriminator().to(device)
    localD = LocalDiscriminator().to(device)
    globalD.apply(weights_init_normal)
    localD.apply(weights_init_normal)
    # print(globalD) # DEBUG only: causes repetitive printing

    # Setup criterion as a dict for multiple loss
    criterion = {
        # WGAN-GP: BCE/BCEWithLogitsLoss avoided to switch to the WGAN-GP
        "l1": nn.L1Loss(), # L1 loss function
        "mse": nn.MSELoss() # MSE loss function
    }

    # Setup optimizers
    optimizer_netG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.0, 0.9))
    optimizer_globalD = optim.Adam(globalD.parameters(), lr=LR_D, betas=(0.0, 0.9))
    optimizer_localD = optim.Adam(localD.parameters(), lr=LR_D, betas=(0.0, 0.9))

    # VGG style and perceptual loss
    style_loss_func = VGG19StyleLoss().to(device)
    perceptual_loss_func = VGG16PerceptualLoss().to(device)

    # Setup gradient scaler
    # Scales losses for mixed precision, hence fast training and low memory usage
    scalerG = amp.GradScaler()
    scaler_globalD = amp.GradScaler()
    scaler_localD = amp.GradScaler()

    # Setup datasets and loaders
    # CroppedImageDataset already applies below transforms
    # (ToTensor, Normalize to [-1, 1], RandomHorizontalFlip, ColorJitter)
    # To use a different transform, create here and pass as a parameter
    train_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'train'), split='train')
    val_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'val'), split='val')
    train_loader = make_dataloader(train_set, 'train', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available)
    val_loader = make_dataloader(val_set, 'val', BATCH_SIZE, NUM_WORKERS, cuda=cuda_available)

    # Setup quality metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Setup fixed samples for visualization
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

    # Setup trackers for batch loss and average epoch loss
    blog = { # batch loss logging
        "totalG": [],
        "totalD": [],
        "globalD": [],
        "localD": []
    }
    elog = { # epoch loss logging
        "totalG": [],
        "totalD": [],
        "globalD": [],
        "localD": [],
        "valG": [] # added for validation
    }

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    print("Starting training...")
    global_step = 0
    start_time = datetime.now()

    for epoch in range(EPOCH_NUM):
        # print(torch.cuda.memory_allocated()) # DEBUG only: check GPU memory
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1} / {EPOCH_NUM}", # Start epoch with 1
            leave=True,
            ncols=100
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Add schedula initializations if necessary

        for i, (masked, original, mask_known) in train_tqdm:
            # Dataset returns batches of: masked, original, mask_known (1=known)
            masked = masked.to(device) # masked image
            original = original.to(device) # original image
            mask_known = mask_known.to(device) # mask (known=1, hole=0)
            mask_hole = (1.0 - mask_known).float().to(device) # (known=0, hole=1)

            # Create a dictionary to keep batch losses
            losses = {}

            # -------------------------------------------------------------------
            # Step 1: Discriminators training as critics (WGAN-GP)
            # -------------------------------------------------------------------
            # Clear out the gradients for tracking
            optimizer_globalD.zero_grad()
            optimizer_localD.zero_grad()

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                fake = netG(original, mask_hole) # full fake image (coarse + fine)
                composite = fake * mask_hole + original * (1.0 - mask_hole) # blend with known

            # Create detached versions to avoid multiple forwarding of generator
            fake_detached = fake.detach()
            composite_detached = fake_detached * mask_hole + original * (1.0 - mask_hole)

            # Update globalD for global critic
            with amp.autocast(device_type='cuda', dtype=torch.float16):
                real_global = globalD(original)
                fake_global = globalD(composite_detached)
            # For stability, gradient penalty HAS to be float32! (no amp)
            gp_global = gradient_penalty(globalD, original, composite_detached, device)
            loss_globalD = (fake_global.mean() - real_global.mean()) + GP_LAMBDA * gp_global

            # Use scaler, DO NOT detach before backward to keep gradients flow
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
            # For stability, gradient penalty HAS to be float32! (no amp)
            gp_local = gradient_penalty(localD, real_patches, fake_patches, device)
            loss_localD = (fake_local.mean() - real_local.mean()) + GP_LAMBDA * gp_local

            # Use scaler, DO NOT detach before backward to keep gradients flow
            scaler_localD.scale(loss_localD).backward() # Scale loss
            scaler_localD.step(optimizer_localD) # skips if unscaled is inf or NaN
            scaler_localD.update() # Updates scale for the next iteration
            losses["localD"] = loss_localD.detach()

            # Final discriminator loss
            losses["totalD"] = (losses["globalD"] + losses["localD"]).detach()

            # Free memory for discriminator step
            del fake_detached, composite_detached, real_global, fake_global, gp_global
            del real_patches, fake_patches, real_local, fake_local, gp_local

            # -------------------------------------------------------------------
            # Step 2: Generator training
            # -------------------------------------------------------------------
            # Clear out the gradients for tracking
            optimizer_netG.zero_grad()

            with amp.autocast(device_type='cuda', dtype=torch.float16):
                # ---------------------------------------------------------------
                # Adversarial loss (negated critic scores)
                # ---------------------------------------------------------------
                adv_global = -globalD(composite).mean()
                patches = crop_local_patch(composite, mask_hole)
                adv_local = -localD(patches).mean()
                losses["adv"] = adv_global + adv_local

                # ---------------------------------------------------------------
                # Pixel-wise L1 loss (multiscale, under amp)
                # ---------------------------------------------------------------
                # Weighted l1 loss: https://arxiv.org/pdf/2401.03395
                # Also weighted: https://arxiv.org/pdf/1801.07892
                # Training only the hole may cause seam artifacts at boundary and inconsistencies
                multi_l1 = 0.0
                # Downsample images to multiple scales
                o_scale = downsample(original, SCALES)
                f_scale = downsample(fake, SCALES)
                m_scale = downsample(mask_hole, SCALES)
                for ors, fs, ms in zip(o_scale, f_scale, m_scale):
                    multi_l1 += HOLE_LAMBDA * criterion["l1"](fs * ms, ors * ms) + \
                                VALID_LAMBDA * criterion["l1"](fs * (1.0 - ms), ors * (1.0 - ms))
                losses["l1"] = multi_l1 / len(SCALES)

            # -------------------------------------------------------------------
            # Style & Perceptual loss (no amp to avoid NaN, only full scale)
            # -------------------------------------------------------------------
            with amp.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                comp_full = (fake * mask_hole + original * (1.0 - mask_hole)).float()
                sl = style_loss_func(original.float(), comp_full)
                pl = perceptual_loss_func(original.float(), comp_full)
            losses["style"] = sl
            losses["perceptual"] = pl

            # DEBUG only: Check for loss values to find the cause of NaN
            if not all(torch.isfinite(x) for x in [losses["adv"], losses["l1"], sl, pl]):
                print(f"[DEBUG][epoch {epoch}] iter {i}"
                      f"adv={float(losses['adv'])}"
                      f"l1={float(losses['l1'])}"
                      f"style={float(losses['style'])}"
                      f"perceptual={float(losses['perceptual'])}")

            # Final generator loss
            losses["totalG"] = (ADV_LAMBDA * losses["adv"] +
                                L1_LAMBDA * losses["l1"] +
                                STYLE_LAMBDA * losses["style"] +
                                PERCEPTUAL_LAMBDA * losses["perceptual"])

            # Use scaler, DO NOT detach before backward to keep gradients flow
            scalerG.scale(losses["totalG"]).backward()
            scalerG.unscale_(optimizer_netG) # Added due to NaN G and for stability
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0) # Clip gradients
            scalerG.step(optimizer_netG)
            scalerG.update()

            # Free memory for generator step
            del adv_local, adv_global, patches, o_scale, f_scale, m_scale

            # -------------------------------------------------------------------
            # Step 3: Batch logging and visualizing
            # -------------------------------------------------------------------
            blog["totalG"].append(losses["totalG"].item())
            blog["totalD"].append(losses["totalD"].item())
            blog["globalD"].append(losses["globalD"].item())
            blog["localD"].append(losses["localD"].item())

            # Print the progress and batch losses (once in every 100 batch)
            if i % SAVE_FREQ == 0:
                train_tqdm.set_postfix({
                    'G': losses["totalG"].item(),
                    'D': losses["totalD"].item(),
                    'gD': losses["globalD"].item(),
                    'lD': losses["localD"].item()
                })

                # Save the current batch (16) images: [original | masked | composite]
                with torch.no_grad():
                    vis_comp = composite # use a clone if the tensor is manipulated later
                    grid = torch.cat( # dimensions: (C,H,W)
                        [to_unit(original), to_unit(masked), to_unit(vis_comp)],
                        3) # horizontal stack per sample using width dimension
                    vutils.save_image(grid,
                                      os.path.join(OUT_PATH, f'comparison_epoch({epoch})_batch({i}).png'),
                                      normalize=False, # done already above with to_unit
                                      nrow=1) # 1 row per sample
            global_step += 1

        # -----------------------------------------------------------------------
        # Epoch logging: Training
        # -----------------------------------------------------------------------
        # Compute and store average losses for the current epoch
        gmean = sum(blog["totalG"][-len(train_loader):]) / len(train_loader)
        dmean = sum(blog["totalD"][-len(train_loader):]) / len(train_loader)
        gdmean = sum(blog["globalD"][-len(train_loader):]) / len(train_loader)
        ldmean = sum(blog["localD"][-len(train_loader):]) / len(train_loader)

        elog["totalG"].append(gmean)
        elog["totalD"].append(dmean)
        elog["globalD"].append(gdmean)
        elog["localD"].append(ldmean)

        # -----------------------------------------------------------------------
        # Validation loop
        # -----------------------------------------------------------------------
        val_losses = []
        with torch.no_grad():
            for masked, original, mask_known in val_loader:
                masked = masked.to(device)
                original = original.to(device)
                mask_known = mask_known.to(device)
                mask_hole = (1.0 - mask_known).float().to(device)

                # Generator forward w/o grad
                fake = netG(original, mask_hole)
                composite = fake * mask_hole + original * (1.0 - mask_hole)

                # Adversarial (negated critic scores)
                with amp.autocast(device_type='cuda', dtype=torch.float16):
                    # -----------------------------------------------------------
                    # Adversarial loss (negated critic scores)
                    # -----------------------------------------------------------
                    adv_global = -globalD(composite).mean()
                    patches = crop_local_patch(composite, mask_hole)
                    adv_local = -localD(patches).mean()
                    adv_loss = adv_global + adv_local

                    # -----------------------------------------------------------
                    # Pixel-wise L1 loss (multiscale, under amp)
                    # -----------------------------------------------------------
                    vl1 = 0.0
                    # Downsample images to multiple scales
                    o_vscale = downsample(original, SCALES)
                    f_vscale = downsample(fake, SCALES)
                    m_vscale = downsample(mask_hole, SCALES)

                    for ors, fs, ms in zip(o_vscale, f_vscale, m_vscale):
                        vl1 += HOLE_LAMBDA * criterion["l1"](fs * ms, ors * ms) + \
                               VALID_LAMBDA * criterion["l1"](fs * (1.0 - ms), ors * (1.0 - ms))
                    l1_loss = vl1 / len(SCALES)

                # ---------------------------------------------------------------
                # Style & Perceptual loss (no amp to avoid NaN, only full scale)
                # ---------------------------------------------------------------
                with amp.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                    comp_full = (fake * mask_hole + original * (1.0 - mask_hole)).float()
                    vsl = style_loss_func(original.float(), comp_full)
                    vpl = perceptual_loss_func(original.float(), comp_full)

                # Final generator loss
                totalG_val_loss = (ADV_LAMBDA * adv_loss +
                                   L1_LAMBDA * l1_loss +
                                   STYLE_LAMBDA * vsl +
                                   PERCEPTUAL_LAMBDA * vpl)
                val_losses.append(totalG_val_loss.item())

        # -----------------------------------------------------------------------
        # Validation logging
        # -----------------------------------------------------------------------
        avg_valG = sum(val_losses) / len(val_losses)
        elog["valG"].append(avg_valG)
        print(f"Validation Epoch: {epoch+1}: G={avg_valG:.4f}")

        # -----------------------------------------------------------------------
        # Save models (Generator and Discriminators)
        # -----------------------------------------------------------------------
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(netG.state_dict(), os.path.join(OUT_PATH, f'netG_epoch{epoch+1}.pth'))
            torch.save(globalD.state_dict(), os.path.join(OUT_PATH, f'globalD_epoch{epoch+1}.pth'))
            torch.save(localD.state_dict(), os.path.join(OUT_PATH, f'localD_epoch{epoch+1}.pth'))

        # -----------------------------------------------------------------------
        # Epoch logging and visualization (fixed samples)
        # -----------------------------------------------------------------------
        with torch.no_grad():
            fixed_fake = netG(fixed_original, fixed_mask_hole).detach()
            fixed_comp = (fixed_fake * fixed_mask_hole + fixed_original * (1.0 - fixed_mask_hole)).detach()

            # Convert images to [0,1] for RGB and visualization
            unit_original = to_unit(fixed_original)
            unit_comp = to_unit(fixed_comp)
            unit_masked = to_unit(fixed_masked)

            # Print quality metrics values for the current epoch
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

            # Save the current batch (16) images: [image1 | image2 | image3 | ...]
            grid = torch.cat(
                [unit_original, unit_masked, unit_comp],
                0) # vertical stack per sample
            vutils.save_image(
                grid,
                os.path.join(OUT_PATH, f'fixed_epoch{epoch+1}.png'),
                normalize=False,
                nrow=fixed_original.size(0)
            )

        # Conditional cleanup for memory (added after doubling epoch time)
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            peak = torch.cuda.max_memory_reserved(0)

            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated()
            fragment = reserved - allocated

            near_oom = peak / total > 0.90
            high_frag = fragment / total > 0.60

            if near_oom or high_frag:
                torch.cuda.empty_cache()

    print("Training complete!")
    stop_time = datetime.now()
    print(f"Time:{(stop_time - start_time)} | Steps:{global_step}")

    # Plot per-batch loss curve
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (per batch)")
    plt.plot(blog["totalG"], label="G (batch)")
    plt.plot(blog["totalD"], label="D (batch)")
    plt.plot(blog["globalD"], label="gD (batch)")
    plt.plot(blog["localD"], label="lD (batch)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_PATH, "loss_curve_batch.png"))
    plt.close()

    # Plot per-epoch averaged loss curve (Like DCGAN)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training (per epoch avg)")
    plt.plot(elog["totalG"], label="G (epoch avg)")
    plt.plot(elog["totalD"], label="D (epoch avg)")
    plt.plot(elog["globalD"], label="gD (epoch avg)")
    plt.plot(elog["localD"], label="lD (epoch avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_PATH, "loss_curve_epoch.png"))
    plt.close()

if __name__ == "__main__":
    # Redirect stdout to log file (keep inside __main__ to avoid multiprocessing errors)
    #print(f"Logging to {LOG_FILE}\n")
    #sys.stdout = utils.StdOut(LOG_FILE)

    # Start training process
    main()