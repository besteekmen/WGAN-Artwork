import os
import random
import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import FiveCrop, RandomCrop
from tqdm import tqdm
from utils.utils import clear_folder, is_cuda, get_device
from config import LOCAL_PATCH_SIZE, DATA_PATH, BATCH_SIZE, NUM_WORKERS


class CroppedImageDataset(Dataset):
    """Dataset of previously cropped images.

    Data is obtained from: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

    A Pytorch dataset to scan pre-extracted cropped images,
    and return the original crop with a (randomly) masked version and the mask
    Attributes: or args:? add returns:
        crops_dir (str): Directory path containing the pre-extracted crops.
        transform (callable, optional): A function/transform that takes in
        a PIL image and returns a transformed version.
        split (str): One of 'train', 'val', 'test'
            - train/val: random masks each epoch
            - test: deterministic mask per index (reproducible)
    Returns:
        Dataset: Cropped dataset of cropped images and masks (crop: [-1,1], mask: [0,1]).
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
            rand = random.Random(index) if self.split == 'test' else None
            square_mask = generate_square_mask(height, width, rand=rand)
            irregular_mask = generate_irregular_mask(height, width, rand=rand)
            mask = torch.cat([square_mask, irregular_mask], dim=0)

            # removed redundant masked_crop, added 2 channel mask [B, 2, H, W]
            return crop, mask

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

def generate_square_mask(height, width, rand=None):
    """Create a random size and random location square mask."""
    rand = rand or random
    block_size = rand.randint(min(height, width) // 4, min(height, width) // 2)
    top = rand.randint(0, height - block_size)
    left = rand.randint(0, width - block_size)
    mask = torch.ones(1, height, width, dtype=torch.float32)  # [1, H, W]
    mask[:, top:top + block_size, left:left + block_size] = 0  # (1 = known, 0 = hole)
    return mask

def generate_irregular_mask(height, width, brush_width=(7, 25),
                            min_times=6, max_times=10,
                            rand=None):
    """Create a random size and random location irregular mask.
    Src: LaMa Image Inpainting WACV 2022 (https://github.com/advimman/lama)"""
    rand = rand or random
    mask = np.ones((height, width), np.float32)  # [H, W] need to be numpy for opencv
    times = rand.randint(min_times, max_times)
    for _ in range(times):
        shape = rand.choice(['line', 'circle'])
        bw = rand.randint(*brush_width)
        if shape == 'circle':
            cx, cy = rand.randint(0, width), rand.randint(0, height)
            cv2.circle(mask, (cx, cy), radius=bw, color=0.0, thickness=-1)
        elif shape == 'line':
            x1, y1 = rand.randint(0, width), rand.randint(0, height)
            x2, y2 = rand.randint(0, width), rand.randint(0, height)
            cv2.line(mask, (x1, y1), (x2, y2), color=0.0, thickness=bw)
    return torch.from_numpy(mask).float().unsqueeze(0) # [1, H, W]

def randomize_masks(mask, irr_ratio=0.3):
    """Randomly select mask types using the given irregular mask ratio.
    Arguments:
        mask: a batch of masks with shape [B, 2, H, W] of [:,0] square, [:,1] irregular.
        irr_ratio: the ratio of irregular masks to random masks.
    Returns:
        mask: a batch of masks of randomized form with shape [B, 1, H, W].
    """
    B = mask.size(0)
    form = torch.zeros(B, dtype=torch.long, device=mask.device)
    irr = round(irr_ratio * B)
    i = torch.randperm(B, device=mask.device)[:irr]
    form[i] = 1
    i = form.view(B, 1, 1, 1).expand(-1, 1, *mask.shape[2:])
    return mask.gather(1, i)

def make_dataloader(dataset, set_path, batch_size, num_workers, cuda, shuffle=True):
    assert len(dataset) > 0, f"No crops found in {set_path} set!"
    # Try high-performance dataloader
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=cuda,
            persistent_workers=True,
            prefetch_factor=2
        )
        _ = next(iter(dataloader))  # force load to test
    except Exception as e:
        print("High-performance dataloader error, falling back to num_workers=0:", e)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    # Add a pin_memory=True argument when calling torch.utils.data.DataLoader()
    # on small datasets, to make sure data is stored at fixed GPU memory addresses
    # and thus increase the data loading speed during training.
    return dataloader

def prepare_dataset(data_path=DATA_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Prepare datasets through dataloaders."""
    train_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'train'), split='train')
    val_set = CroppedImageDataset(crops_dir=os.path.join(DATA_PATH, 'val'), split='val')
    train_loader = make_dataloader(train_set, 'train', BATCH_SIZE, NUM_WORKERS, cuda=is_cuda())
    val_loader = make_dataloader(val_set, 'val', BATCH_SIZE, NUM_WORKERS, cuda=is_cuda())
    return train_loader, val_loader

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
                    save_path = os.path.join(target_dir, save_name)
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