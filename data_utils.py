import os
import random
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import FiveCrop, RandomCrop
from tqdm import tqdm
from utils import clear_folder
from config import LOCAL_PATCH_SIZE

class CroppedImageDataset(Dataset):
    """
    A Pytorch dataset to scan pre-extracted cropped images,
    and return the original crop with a (randomly) masked version and the mask

    Args:
        crops_dir (str): Directory path containing the pre-extracted crops.
        transform (callable, optional): A function/transform that takes in
        a PIL image and returns a transformed version.
        split (str): One of 'train', 'val', 'test'
            - train/val: random masks each epoch
            - test: deterministic mask per index (reproducible)
    """
    def __init__(self, crops_dir, transform=None, split='train'):
        self.crops_dir = crops_dir
        self.split = split.lower()
        assert self.split in ['train', 'val', 'test'], f"Split {split} not recognized.!"
        self.transform = transform or transforms.Compose([
            # to extend further transformations, add below lines
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            # GANs perform better when inputs are in [-1,1] range instead of [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.crop_paths = self._scan_crops(crops_dir)

    def __len__(self):
        return len(self.crop_paths)

    def __getitem__(self, index):
        crop_path = self.crop_paths[index]

        try:
            image = Image.open(crop_path).convert('RGB')
            crop = self.transform(image)

            # Create a binary mask
            _, height, width = crop.shape

            if self.split == 'test':
                # Deterministic mask generation using index
                rndx = random.Random(index)
                block_size = rndx.randint(min(height, width) // 4, min(height, width) // 2)
                top = rndx.randint(0, height - block_size)
                left = rndx.randint(0, width - block_size)
            else:
                # Random mask at each call
                block_size = random.randint(min(height, width) // 4, min(height, width) // 2)
                top = random.randint(0, height - block_size)
                left = random.randint(0, width - block_size)

            mask = torch.ones(1, height, width, dtype=torch.float32) # [1, H, W]
            mask[:, top:top + block_size, left:left + block_size] = 0
            # due to automatic broadcast, below works despite the channel mismatch
            masked_crop = crop * mask

            # mask here is human-friendly version, need to be converted for generator
            # masked_crop: [-1,1], crop: [-1,1], mask: [0,1]
            return masked_crop, crop, mask

        except Exception as e:
            print(f"Error loading ({crop_path}): {e}.")
            # Try next image to avoid a crash
            next_index = (index + 1) % len(self.crop_paths)
            if next_index == index:
                raise RuntimeError(f"All image loading failed at index {index}.")
            return self.__getitem__(next_index)

    @staticmethod
    def _scan_crops(root_dir):
        """
        Scans directory for cropped images.
        """
        # TODO: add image extension check later!
        #image_ext = {'.jpg', '.jpeg', '.png'}
        all_crops = []
        print(f"Scanning for crops in: {root_dir} ...")

        for root, _, files in os.walk(root_dir):
            for filename in tqdm(files, desc=f"Scanning {os.path.basename(root)}", leave=False):
                ext = os.path.splitext(filename)[1].lower()
                #if ext in image_ext and '_crop' in filename:
                if '_crop' in filename:
                    all_crops.append(os.path.join(root, filename))

        print(f"Found {len(all_crops)} crop images.")
        return all_crops

def make_dataloader(dataset, set_path, batch_size, num_workers, cuda, shuffle=True):
    """Optimized dataloader for the current hardware (RTX 4060 + i7-13620H).
    Tiered fallback applied."""
    assert len(dataset) > 0, f"No crops found in {set_path} set!"

    # Set num_workers if not provided
    if num_workers is None:
        num_workers = 8 if shuffle else 4 # more for training and less for validation

    # Try high-performance dataloader from the set of fallbacks
    for workers in [8, 4, 0]:
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle, # randomly shuffle dataset at each epoch
                num_workers=workers,
                pin_memory=cuda,
                prefetch_factor=2 if workers > 0 else 1,
                # load 2 batches ahead per worker to reduce GPU waiting time
                persistent_workers=True if num_workers > 0 else False, # workers stay alive to avoid restarting overhead
                drop_last=True if shuffle else False # ensure all batches are full-size
            )
            # Test load to catch issues earlier
            _ = next(iter(dataloader))
            print(f"Using num_workers={workers} ...")
            return dataloader
        except Exception as e:
            print(f"Dataloader with num_workers={workers} failed: {e}.")
    # Add a pin_memory=True argument when calling torch.utils.data.DataLoader()
    # on small datasets, to make sure data is stored at fixed GPU memory addresses
    # and thus increase the data loading speed during training.

    # If all fails, raise an error
    raise RuntimeError(f"Dataloader failed with all fallback options (8, 4, 0).")

def preextract_fivecrops(source_dir, target_dir, crop_size=LOCAL_PATCH_SIZE):
    """
    Extract five fixed size crops (center + 4 corners) from each image,
    then save them all as separate images.

    Args:
        source_dir (str): Directory path containing the images to crop.
        target_dir (str): Directory path where the extracted crops will be saved.
        crop_size (int): Size of the crop to extract.
    """
    cropper = FiveCrop(crop_size)
    image_ext = {'.jpg', '.jpeg', '.png'}

    #os.makedirs(target_dir, exist_ok=True)
    clear_folder(target_dir)
    print(f"Extracting five crops from images in ({source_dir}) to ({target_dir}).")

    for root, _, files in os.walk(source_dir):
        # Skip folders starting with "_"
        if any(folder.startswith('_') for folder in os.path.relpath(root, source_dir).split(os.sep)):
            continue

        for filename in tqdm(files, desc=f"Scanning {os.path.basename(root)}", leave=False):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_ext:
                continue

            image_path = os.path.join(root, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                crops = cropper(image)

                base_name = os.path.splitext(filename)[0]
                for i, crop in enumerate(crops):
                    save_name = f"{base_name}_crop{i}.png"
                    save_patha = os.path.join(target_dir, save_name)
                    crop.save(save_path)

            except Exception as e:
                print(f"Failed processing ({image_path}): {e}.")

    print(f"Finished extracting all crops.")

def preextract_randomcrops(source_dir, target_dir, crop_size=LOCAL_PATCH_SIZE, crops_per_image=3):
    """
    Extract a specified number of fixed size crops (all located randomly) from each image,
    then save them all as separate images.

    Args:
        source_dir (str): Directory path containing the images to crop.
        target_dir (str): Directory path where the extracted crops will be saved.
        crop_size (int): Size of the crop to extract.
        crops_per_image (int): Number of crops to extract.
    """
    cropper = RandomCrop(crop_size)
    image_ext = {'.jpg', '.jpeg', '.png'}

    os.makedirs(target_dir, exist_ok=True)
    #TODO: clear_folder(target_dir)
    print(f"Extracting ({crops_per_image}) random crops from images in ({source_dir}) to ({target_dir}).")

    for root, _, files in os.walk(source_dir):
        # Skip folders starting with "_"
        if any(folder.startswith('_') for folder in os.path.relpath(root, source_dir).split(os.sep)):
            continue

        for filename in tqdm(files, desc=f"Scanning {os.path.basename(root)}", leave=False):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_ext:
                continue

            image_path = os.path.join(root, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                width, height = image.size

                # Skip small images
                if width < crop_size or height < crop_size:
                    print(f"Skipping small image ({width}x{height}): ({image_path}).")
                    continue

                base_name = os.path.splitext(filename)[0]
                for i in range(crops_per_image):
                    crop = cropper(image)
                    save_name = f"{base_name}_crop{i}.png"
                    save_path = os.path.join(target_dir, save_name)
                    crop.save(save_path)

            except Exception as e:
                print(f"Failed processing ({image_path}): {e}.")

    print(f"Finished extracting all crops.")