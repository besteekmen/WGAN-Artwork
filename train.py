import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from datetime import datetime
from tqdm import tqdm
from multiprocessing import freeze_support
from torch import amp

# Project specific modules
from config import *
from losses import gradient_penalty, init_losses, lossMSL1, init_metrics, lossEdge
from models.model_builder import init_optimizers, init_nets, save_checkpoint, setup_model, forward_pass
from utils.utils import to_unit, set_seed, get_device, print_device, make_run_directory, half_precision, \
    full_precision, get_schedule, set_logger, is_cuda, clamp_f32
from dataset import prepare_dataset, prepare_batch
from utils.vision_utils import crop_local_patch, plot_loss, set_fixed, save_images

# ------------------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------------------
def main():
    freeze_support() # for Windows multiprocessing

    # Create training directories and logger
    out_path, log_path, check_path = make_run_directory()
    logger = set_logger(log_path)

    # Setup seed for randomness (if not pre-defined)
    print(f"PyTorch version: {torch.__version__}")
    print_device()
    set_seed(SEED)
    device = get_device()

    # CUDNN setups
    cudnn.benchmark = True
    # Set to false if strange OOM (Out of memory) issues seen
    # cudnn.deterministic = True
    # Set to true if exact reproducibility is needed and disable benchmark

    # Initialize or load the model
    netG, globalD, localD = init_nets(device)
    optimG, optimGD, optimLD = init_optimizers(netG, globalD, localD)
    start_epoch = setup_model(netG, globalD, localD,
                              optimG, optimGD, optimLD, check_path, device)
    # To load a pretrained model, add file_name as parameter to setup_model

    # Setup losses and quality metrics
    lossStyle, lossPerceptual = init_losses()
    ssim, lpips, fid = init_metrics()

    # Setup gradient scaler (provide mixed precision for faster training and low memory usage)
    scalerG = amp.GradScaler(enabled=is_cuda())
    scaler_globalD = amp.GradScaler(enabled=is_cuda())
    scaler_localD = amp.GradScaler(enabled=is_cuda())

    # Setup datasets loaders
    train_loader, val_loader = prepare_dataset()

    # Setup fixed samples for visualization
    fixed_masked, fixed_image, fixed_mask_hole = set_fixed(train_loader.dataset, device=device)

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
    logger.info("Starting training...")
    global_step = 0
    start_time = datetime.now()
    tolerance = 5
    best_fid = float("inf")
    early_stopping = False

    for epoch in range(start_epoch, EPOCH_NUM):
        # print(torch.cuda.memory_allocated()) # DEBUG only: check GPU memory
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1} / {EPOCH_NUM}", # Start epoch with 1
            leave=False,
            ncols=100
        )

        nan_log_i = -999999

        # Set lambda schedules
        adv_lambda = get_schedule(epoch, ADV_LAMBDA_SCHEDULE)
        perc_lambda = get_schedule(epoch, PERCEPTUAL_LAMBDA_SCHEDULE)
        style_lambda = get_schedule(epoch, STYLE_LAMBDA_SCHEDULE)
        edge_lambda = get_schedule(epoch, EDGE_LAMBDA_SCHEDULE)
        irr_ratio = get_schedule(epoch, IRR_RATIO_SCHEDULE)

        # Set training mode
        netG.train(); globalD.train(); localD.train()

        # Initialize loss sums
        g_tot, d_tot, gd_tot, ld_tot = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for i, (image, mask) in train_tqdm:
            image, mask_hole = prepare_batch((image, mask), device,
                                                   irr_ratio = irr_ratio,
                                                   non_blocking = True)

            # Create a dictionary to keep batch losses
            losses = {}

            # -------------------------------------------------------------------
            # Step 1: Discriminators training as critics (WGAN-GP)
            # -------------------------------------------------------------------
            # Clear out the gradients for tracking
            optimGD.zero_grad()
            optimLD.zero_grad()

            with half_precision():
                fake, composite = forward_pass(netG, image, mask_hole)
            # Create detached versions to avoid multiple forwarding of generator
            composite_detached = composite.detach()

            # Update globalD for global critic
            with half_precision():
                real_global = globalD(image)
                fake_global = globalD(composite_detached)
            # For stability, gradient penalty HAS to be float32! (no amp)
            gp_global = gradient_penalty(globalD, image.float(), composite_detached.float(), device)
            loss_globalD = (fake_global.mean() - real_global.mean()) + GP_LAMBDA * gp_global

            # Use scaler, DO NOT detach before backward to keep gradients flow
            scaler_globalD.scale(loss_globalD).backward()
            scaler_globalD.unscale_(optimGD)  # Added due to NaN G and for stability
            torch.nn.utils.clip_grad_norm_(globalD.parameters(), 1.0)  # Clip gradients
            scaler_globalD.step(optimGD)
            scaler_globalD.update()
            losses["globalD"] = loss_globalD.detach()

            # Update localD for local critic
            # Using detached versions for discriminator is okay, but not okay for generator
            real_patches = crop_local_patch(image, mask_hole)
            fake_patches = crop_local_patch(composite_detached, mask_hole)
            with half_precision():
                real_local = localD(real_patches)
                fake_local = localD(fake_patches)
            # For stability, gradient penalty HAS to be float32! (no amp)
            gp_local = gradient_penalty(localD, real_patches.float(), fake_patches.float(), device)
            loss_localD = (fake_local.mean() - real_local.mean()) + GP_LAMBDA * gp_local

            # Use scaler, DO NOT detach before backward to keep gradients flow
            scaler_localD.scale(loss_localD).backward() # Scale loss
            scaler_localD.unscale_(optimLD)  # Added due to NaN G and for stability
            torch.nn.utils.clip_grad_norm_(localD.parameters(), 1.0)  # Clip gradients
            scaler_localD.step(optimLD) # skips if unscaled is inf or NaN
            scaler_localD.update() # Updates scale for the next iteration
            losses["localD"] = loss_localD.detach()

            # Final discriminator loss
            losses["totalD"] = (losses["globalD"] + losses["localD"]).detach()

            # Free memory for discriminator step
            del composite_detached, real_global, fake_global, gp_global
            del real_patches, fake_patches, real_local, fake_local, gp_local

            # -------------------------------------------------------------------
            # Step 2: Generator training
            # -------------------------------------------------------------------
            # Clear out the gradients for tracking
            optimG.zero_grad()

            with half_precision():
                # Adversarial loss (negated critic scores)
                adv_global = -globalD(composite).mean()
                patches = crop_local_patch(composite, mask_hole)
                adv_local = -localD(patches).mean()
                losses["adv"] = adv_global + adv_local

                # Pixel-wise L1 loss (multiscale, under amp)
                losses["l1"] = lossMSL1(image, fake, mask_hole)

                # Edge loss
                losses["edge"] = lossEdge(image, fake)

            # Style & Perceptual loss (no amp to avoid NaN, only full scale)
            with full_precision():
                comp_full = clamp_f32(composite) # reused composite
                orig_full = clamp_f32(image)
                sl = lossStyle(orig_full, comp_full)
                pl = lossPerceptual(orig_full, comp_full)
            losses["style"] = sl
            losses["perceptual"] = pl

            # DEBUG only: Check for loss values to find the cause of NaN
            all_terms = [losses["adv"], losses["l1"], losses["edge"], sl, pl]
            with_nan = (not torch.isfinite(fake).all()) or (not all(torch.isfinite(x) for x in all_terms))

            if with_nan:
                if i - nan_log_i >= 100:
                    message_skip = (f"[SKIP][epoch {epoch + 1}] iter {i} | "
                                    f"adv={float(losses['adv']) if torch.isfinite(losses['adv']) else 'NaN'} "
                                    f"l1={float(losses['l1']) if torch.isfinite(losses['l1']) else 'NaN'} "
                                    f"edge={float(losses['edge']) if torch.isfinite(losses['edge']) else 'NaN'} "
                                    f"style={float(losses['style']) if torch.isfinite(losses['style']) else 'NaN'} "
                                    f"perceptual={float(losses['perceptual']) if torch.isfinite(losses['perceptual']) else 'NaN'}")
                    train_tqdm.write(message_skip)
                    logger.info(message_skip)
                    nan_log_i = i
                optimG.zero_grad(set_to_none=True)
                optimGD.zero_grad(set_to_none=True)
                optimLD.zero_grad(set_to_none=True)
                continue

            # Final generator loss
            losses["totalG"] = (adv_lambda * losses["adv"] +
                                L1_LAMBDA * losses["l1"] +
                                edge_lambda * losses["edge"] +
                                style_lambda * losses["style"] +
                                perc_lambda * losses["perceptual"])

            g_tot += losses["totalG"].item()
            d_tot += losses["totalD"].item()
            gd_tot += losses["globalD"].item()
            ld_tot += losses["localD"].item()
            num_batches += 1

            if i % SAVE_FREQ == 0:
                raw = {k: float(losses[k]) for k in ["adv", "l1", "edge", "style", "perceptual"]}
                weighted = {
                    "adv_w": adv_lambda * raw["adv"],
                    "l1_w": L1_LAMBDA * raw["l1"],
                    "edge_w": edge_lambda * raw["edge"],
                    "style_w": style_lambda * raw["style"],
                    "perceptual_w": perc_lambda * raw["perceptual"]
                }
                logger.info(f"[Debug] Raw: {raw} | Weighted: {weighted}")

            # Use scaler, DO NOT detach before backward to keep gradients flow
            scalerG.scale(losses["totalG"]).backward()
            scalerG.unscale_(optimG) # Added due to NaN G and for stability
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0) # Clip gradients
            scalerG.step(optimG)
            scalerG.update()

            # Free memory for generator step
            del adv_local, adv_global, patches
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

                # Save the current batch (16) images: [image | masked | composite]
                with torch.no_grad():
                    masked = image * (1.0 - mask_hole)
                    save_images(to_unit(image), to_unit(masked), to_unit(composite), 3,  # horizontal stack per sample using width dimension
                                1, out_path, f'comparison_epoch({epoch})_batch({i}).png')
            global_step += 1

        # -----------------------------------------------------------------------
        # Epoch logging: Training
        # -----------------------------------------------------------------------
        # Compute and store average losses for the current epoch
        gmean = g_tot / max(1, num_batches)
        dmean = d_tot / max(1, num_batches)
        gdmean = gd_tot / max(1, num_batches)
        ldmean = ld_tot / max(1, num_batches)

        elog["totalG"].append(gmean)
        elog["totalD"].append(dmean)
        elog["globalD"].append(gdmean)
        elog["localD"].append(ldmean)

        # -----------------------------------------------------------------------
        # Validation loop
        # -----------------------------------------------------------------------
        # Set validation mode
        netG.eval(); globalD.eval(); localD.eval()

        with torch.no_grad():
            ssim_tot, lpips_tot = 0.0, 0.0
            l1_tot, edge_tot, style_tot, perc_tot = 0.0, 0.0, 0.0, 0.0
            val_batches = 0

            for image, mask in val_loader:
                # only use square masks for validation
                image, mask_hole = prepare_batch((image, mask), device,
                                                       non_blocking=True)

                # Generator forward w/o grad
                fake, composite = forward_pass(netG, image, mask_hole)

                # Compute square mask metrics
                unit_image = to_unit(image)
                unit_comp = to_unit(composite)
                ssim_tot += ssim(unit_comp, unit_image).item()
                lpips_tot += lpips(unit_comp, unit_image).item()
                val_batches += 1

                with half_precision(): # no adv loss for validation!
                    # Pixel-wise L1 loss (multiscale, under amp)
                    l1_loss = lossMSL1(image, fake, mask_hole)

                    # Edge loss
                    edge_loss = lossEdge(image, fake)

                # Style & Perceptual loss (no amp to avoid NaN, only full scale)
                with full_precision():
                    comp_full = clamp_f32(composite) # reused composite
                    orig_full = clamp_f32(image)
                    vsl = lossStyle(orig_full, comp_full)
                    vpl = lossPerceptual(orig_full, comp_full)

                # Loss totals for averaging
                l1_tot += l1_loss.item()
                edge_tot += edge_loss.item()
                style_tot += vsl.item()
                perc_tot += vpl.item()

        # -----------------------------------------------------------------------
        # Validation logging
        # -----------------------------------------------------------------------
        avg_l1 = l1_tot / val_batches
        avg_edge = edge_tot / val_batches
        avg_style = style_tot / val_batches
        avg_perc = perc_tot / val_batches

        val_g = (
            L1_LAMBDA * avg_l1 +
            edge_lambda * avg_edge +
            style_lambda * avg_style +
            perc_lambda * avg_perc
        )

        avg_val_ssim = ssim_tot / val_batches
        avg_val_lpips = lpips_tot / val_batches
        elog["valG"].append(val_g)

        logger.info( # Validation logs
            f"Validation [Square-only] Epoch {epoch + 1}/{EPOCH_NUM} | "
            f"G={val_g:.4f}, "
            f"L1={avg_l1:.4f}, "
            f"Edge={avg_edge:.4f}, "
            f"Style={avg_style:.4f}, "
            f"Perc={avg_perc:.4f} | "
            f"SSIM={avg_val_ssim:.4f}, "
            f"LPIPS={avg_val_lpips:.4f}"
        )

        logger.info( # Training logs
            f"Training Epoch {epoch+1}/{EPOCH_NUM} | "
            f"G={elog['totalG'][-1]:.4f}, "
            f"D={elog['totalD'][-1]:.4f}, "
            f"gD={elog['globalD'][-1]:.4f}, "
            f"lD={elog['localD'][-1]:.4f}"
        )

        # -----------------------------------------------------------------------
        # Save models (Generator and Discriminators)
        # -----------------------------------------------------------------------
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(epoch, netG, globalD, localD, optimG, optimGD, optimLD, check_path)

        # -----------------------------------------------------------------------
        # Epoch logging and visualization (fixed samples)
        # -------------------------------------------- ---------------------------
        with torch.no_grad():
            fixed_fake, fixed_comp = forward_pass(netG, fixed_image, fixed_mask_hole)

            # Convert images to [0,1] for RGB and visualization
            unit_image = to_unit(fixed_image)
            unit_comp = to_unit(fixed_comp)
            unit_masked = to_unit(fixed_masked)

            # Print quality metrics values for the current epoch
            fixed_ssim = ssim(unit_comp, unit_image).item()
            fixed_lpips = lpips(unit_comp, unit_image).item()
            fixed_fid = None
            if (epoch + 1) % 5 == 0:
                fid.reset()
                fid.update(unit_image, real=True)
                fid.update(unit_comp, real=False)
                fixed_fid = fid.compute().item()

                if fixed_fid < best_fid:
                    best_fid = fixed_fid
                    tolerance = 5 # reset if improved
                else:
                    tolerance -= 1
                    if tolerance <= 0:
                        print(f"FID STOP: Early stopping at epoch {epoch+1}/{EPOCH_NUM}")
                        logger.info(f"FID STOP: Early stopping at epoch {epoch+1}/{EPOCH_NUM}")
                        early_stopping = True

            message = f"Epoch {epoch+1}: SSIM={fixed_ssim:.4f} LPIPS={fixed_lpips:.4f}"
            if fixed_fid is not None:
                message += f", FID={fixed_fid:.2f}"
            print(message)
            logger.info(message)

            # Save the current batch (16) images: [image1 | image2 | image3 | ...]
            save_images(unit_image, unit_masked, unit_comp, 0, # vertical stack per sample
                        fixed_image.size(0), out_path, f'fixed_epoch{epoch+1}.png')

        if early_stopping:
            break

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

    # Plot per-batch and per-epoch loss curve
    plot_loss("batch", blog, out_path)
    plot_loss("epoch", elog, out_path) # per epoch average

if __name__ == "__main__":

    # Start training process
    main()