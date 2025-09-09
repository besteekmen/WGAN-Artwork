import os
import shutil
import matplotlib.pyplot as plt
import random
import numpy as np
import torch

from torchvision import transforms
from tqdm import tqdm
from config import SOURCE_PATH, DATA_PATH, CROP_PATH, SAMPLE_PATH, SEED
from utils.utils import clear_folder
from dataset import CroppedImageDataset

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def split_dataset(source, target, ratios=(0.8, 0.1, 0.1), seed=SEED):
    """
    Split all crop images into train/val/test subfolders
    source (str): Folder containing all crop images
    target (str): Folder where to save the split data
    ratios (tuple): List of ratios to split the dataset (must sum to 1)
    seed (int): Random seed
    """
    assert abs(sum(ratios)) <= 1, f"ratios must sum to 1, got {sum(ratios)}"

    all_crops = [os.path.join(source, f) for f in os.listdir(source)]
    random.seed(seed)
    random.shuffle(all_crops)

    n = len(all_crops)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        'train': all_crops[:n_train],
        'val': all_crops[n_train:n_train+n_val],
        'test': all_crops[n_train+n_val:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(target, split)
        os.makedirs(split_dir, exist_ok=True)
        for f in tqdm(files, desc=f"Moving {split}"):
            shutil.move(f, os.path.join(split_dir, os.path.basename(f)))

    print(f"Done! Train={len(splits['train'])} Val={len(splits['val'])} Test={len(splits['test'])}")

def display_samples(crops_dataset, num_samples=5, save=False, save_dir=None):
    """Display some sample crops and their masks."""
    if save and (save_dir is not None):
        clear_folder(save_dir)

    for i in range(num_samples):
        masked, original, mask = crops_dataset[i]

        # Convert tensors to numpy for plotting
        masked_image = masked.permute(1, 2, 0).cpu().numpy()
        original_image = original.permute(1, 2, 0).cpu().numpy()
        mask_image = mask.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(original_image)
        ax[0].set_title("Original Crop")
        ax[0].axis('off')

        ax[1].imshow(mask_image[..., 0], cmap='gray') # also mask_image[:, :, 0] valid
        ax[1].set_title("Mask")
        ax[1].axis('off')

        ax[2].imshow(masked_image)
        ax[2].set_title("Masked Crop")
        ax[2].axis('off')

        if save:
            save_path = os.path.join(save_dir, f"sample_{i}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close() # Free memory
        else:
            plt.show()

if __name__ == "__main__":
    # Extract crops (run once when the raw data is updated)
    source_dir = '../'+SOURCE_PATH
    data_dir = '../'+DATA_PATH
    target_dir = '../'+CROP_PATH
    save_dir = '../'+SAMPLE_PATH
    train_dir = data_dir+'/train'
    #preextract_fivecrops(source_dir, target_dir, CROP_SIZE)
    #preextract_randomcrops(source_dir, target_dir, CROP_SIZE, CROP_COUNT)
    split_dataset(source=target_dir, target=data_dir)

    # Load dataset and show some samples
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CroppedImageDataset(crops_dir=train_dir, transform=transform, split='train')
    display_samples(dataset, num_samples=10, save=True, save_dir=save_dir)



