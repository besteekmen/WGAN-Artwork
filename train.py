import os
# import random (to not use np.random instead)
import sys
from os import environ

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from multiprocessing import freeze_support

from config import *
from utils import clear_folder
from models.generator import Generator
from models.discriminator import Discriminator
from models.weights_init import weights_init
os.environ["PYTHONUNBUFFERED"] = "1"

def main():
    freeze_support() # for Windows multiprocessing

    # Setup out folder and logging
    clear_folder(OUT_PATH)

    # Setup seed for randomness (if not pre-defined)
    print(f"PyTorch version: {torch.__version__}")
    seed_val = np.random.randint(1, 10000) if seed is None else seed
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
    net_g = Generator().to(device)
    net_g.apply(weights_init)
    # print(net_g) # causes repetitive printing

    # Create a Discriminator network object
    net_d = Discriminator().to(device)
    net_d.apply(weights_init)
    # print(net_d) # causes repetitive printing

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss function
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(0.5, 0.999))

    # Load the dataset
    dataset = dset.MNIST(root=DATA_PATH, download=True,
                         transform=transforms.Compose([transforms.Resize(X_DIM),
                                                       transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
                                                       ]))

    # TODO (Another data): Change accordingly to use another data (colored image data)
    # dataset = dset.ImageFolder(root=DATA_PATH,
    #                            transform=transforms.Compose([
    #                                transforms.Resize(X_DIM),
    #                                transforms.CenterCrop(X_DIM),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                            ]))

    assert dataset
    # Try high-performance dataloader
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
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

    # Create empty lists to track loss values
    g_losses = []
    d_losses = []

    # Fixed noise vector for visualization
    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    # Training loop
    for epoch in range(EPOCH_NUM):
        dataloader_tqdm = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1} / {EPOCH_NUM}",
            leave=True,
            ncols=100
        )
        for i, data in dataloader_tqdm:
            x_real = data[0].to(device)
            # Create real and fake label tensors in real time,
            # because there is no guarantee that all sample batches will have the same size
            # BCELoss (Binary Cross Entropy) operates on probabilities,
            # it expects both inputs and targets to be floating-point numbers in the range [0.0, 1.0].
            real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device, dtype=torch.float)
            fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device=device, dtype=torch.float)

            # Step 1: Update D with real data
            net_d.zero_grad()
            y_real = net_d(x_real)
            loss_d_real = criterion(y_real, real_label)
            loss_d_real.backward()

            # Step 2: Update D with fake data
            z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
            x_fake = net_g(z_noise)
            y_fake = net_d(x_fake.detach())
            loss_d_fake = criterion(y_fake, fake_label)
            loss_d_fake.backward()
            optimizer_d.step()

            # Step 3: Update G with fake data
            net_g.zero_grad()
            y_fake_r = net_d(x_fake)
            loss_g = criterion(y_fake_r, real_label)
            loss_g.backward()
            optimizer_g.step()

            # Print the progress
            if i % 100 == 0:
                dataloader_tqdm.set_postfix({
                    'loss_d_real': loss_d_real.item(),
                    'loss_d_fake': loss_d_fake.item(),
                    'loss_g': loss_g.item()
                })
                # print(f'Epoch {epoch} [{i}/{len(dataloader)}]'
                #      f'loss_d_real: {loss_d_real.mean().item():.4f}'
                #      f'loss_d_fake: {loss_d_fake.mean().item():.4f}'
                #      f'loss_g: {loss_g.mean().item():.4f}', flush=True)

                # Add loss values to graph
                g_losses.append(loss_g.item())
                d_losses.append((loss_d_real + loss_d_fake).item())

                vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
                with torch.no_grad():
                    viz_sample = net_g(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(OUT_PATH, f'fake_samples_epoch{epoch}_batch{i}.png'),
                                      normalize=True)

        # Save Generator and Discriminator
        torch.save(net_g.state_dict(), os.path.join(OUT_PATH, f'net_g_{epoch}.pth'))
        torch.save(net_d.state_dict(), os.path.join(OUT_PATH, f'net_d_{epoch}.pth'))

    print('Training complete!')

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUT_PATH, "loss_curve.png"))

# Train Generator and Discriminator networks
if __name__ == "__main__":
    # Redirect stdout to log file (keep inside __main__ to avoid multiprocessing errors)
    #print(f"Logging to {LOG_FILE}\n")
    #sys.stdout = utils.StdOut(LOG_FILE)
    main()