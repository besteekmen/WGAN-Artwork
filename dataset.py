import os
import random
import numpy as np
import torch
import cv2

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import RandomCrop
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from tqdm import tqdm
from utils.utils import is_cuda, clear_folder
from config import DATA_PATH, BATCH_SIZE, NUM_WORKERS, CROP_SIZE, SEED, EPS


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
        all_crops = []
        print(f"Scanning for crops in: {root_dir} ...")

        for root, _, files in os.walk(root_dir):
            for filename in tqdm(files, desc=f"Scanning {os.path.basename(root)}", leave=False):
                if '_crop' in filename:
                    all_crops.append(os.path.join(root, filename))

        print(f"Found {len(all_crops)} crop images.")
        return all_crops

def generate_square_mask(height, width, rand=None,
                         p_rotate = 0.5, max_angle = 30.0):
    """Create a random size and random location square mask."""
    rand = rand or random
    # %60 small/medium and %40 large
    if rand.random() < 0.6:
        low, high = min(height, width) // 6, min(height, width) // 3
    else:
        low, high = min(height, width) // 3, min(height, width) // 2

    block_size = rand.randint(low, high)
    top = rand.randint(0, height - block_size)
    left = rand.randint(0, width - block_size)

    mask = torch.ones(1, height, width, dtype=torch.float32)  # [1, H, W]
    mask[:, top:top + block_size, left:left + block_size] = 0  # (1 = known, 0 = hole)

    if rand.random() < p_rotate:
        angle = rand.uniform(-max_angle, max_angle)
        mask = rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=[1.0]
        )
        mask = (mask > 0.5).float()
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
    batch_size = mask.size(0)
    form = torch.zeros(batch_size, dtype=torch.long, device=mask.device)
    irr = round(irr_ratio * batch_size)
    i = torch.randperm(batch_size, device=mask.device)[:irr]
    form[i] = 1
    i = form.view(batch_size, 1, 1, 1).expand(-1, 1, *mask.shape[2:])
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
    train_set = CroppedImageDataset(crops_dir=os.path.join(data_path, 'train'), split='train')
    val_set = CroppedImageDataset(crops_dir=os.path.join(data_path, 'val'), split='val')
    train_loader = make_dataloader(train_set, 'train', batch_size, num_workers, cuda=is_cuda())
    val_loader = make_dataloader(val_set, 'val', batch_size, num_workers, cuda=is_cuda())
    return train_loader, val_loader

def prepare_batch(batch, device, irr_ratio: float | None = None,
                  non_blocking: bool = True):
    """Prepare the given batch of images by setting device and mask selection."""
    image, mask = batch # Dataset returns batches of: image, mask (1=known, 0=hole)
    image = image.to(device, non_blocking=non_blocking)  # ground truth
    mask = mask.to(device, non_blocking=non_blocking)  # masks, [B, 2, H, W] (1=known, 0=hole)

    if irr_ratio is not None:
        mask = randomize_masks(mask, irr_ratio)
    else:
        mask = mask[:, 0:1, :, :] # no randomization, only square masks
    mask_hole = (1.0 - mask).float()  # (1=hole, 0=known)
    return image, mask_hole

def random_select(source_dir, count, size=CROP_SIZE, seed=SEED):
    """Select given count of images from a source directory randomly."""
    rand = random.Random(seed)
    image_ext = {'.jpg', '.jpeg', '.png'}
    base = os.path.basename(source_dir)
    images = []

    for image in tqdm(os.scandir(source_dir), desc=f"Scanning {base}", leave=False):
        if not image.is_file():
            continue
        ext = os.path.splitext(image.name)[1].lower()
        if ext not in image_ext:
            continue
        try:
            with Image.open(image.path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size

            # Skip small images
            if width < size or height < size:
                print(f"Skipping small image ({width}x{height}): ({image.path}).")
            else:
                images.append(image.path)
        except Exception as e:
            print(f"Failed processing ({image.path}): {e}.")

    rand.shuffle(images)
    if len(images) < count:
        print(f"[WARN] Category {base}: requested ({count}), available ({len(images)}).")
    return images[:min(count, len(images))]

def get_crops(image_paths, crop_size=CROP_SIZE, crops_per_image=3):
    """Cuts specified number of crops from the images in given directory."""
    cropper = RandomCrop(crop_size) # change here for reproducible crops
    crops = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert('RGB')
                width, height = img.size

                # Skip small images
                if width < crop_size or height < crop_size:
                    print(f"Skipping small image ({width}x{height}): ({image_path}).")
                    continue

                base = os.path.splitext(os.path.basename(image_path))[0]
                for i in range(crops_per_image):
                    crop = cropper(img)
                    crops.append((crop, base, i))

        except Exception as e:
            print(f"Failed processing ({image_path}): {e}.")
    return crops

def split_data(images, target_dir,
               ratios=(0.8, 0.1, 0.1), prefix=""):
    """
    Split a list of images into train/val/test and save them.
    images (list): List of tuples (image, base, index)
    target_dir (str): Folder where to save the split data
    ratios (tuple): Ratios to split data into
    seed (int): Random seed
    prefix (str): Identifier for filenames
    """
    assert abs(sum(ratios) - 1.0) < EPS, f"ratios must sum to 1, got {sum(ratios)}"
    if not images:
        return 0, 0, 0

    for split in ("train", "val", "test"):
        out_dir = os.path.join(target_dir, split)
        if not os.path.exists(out_dir):
            raise FileNotFoundError(f"{out_dir} does not exist.")

    # no need for randomization here, done in random_select
    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    pfx = (prefix + "_") if prefix else ""
    counts = {"train": 0, "val": 0, "test": 0}

    for split, files in splits.items():
        split_dir = os.path.join(target_dir, split)
        for img, base, i in tqdm(files, desc=f"Saving {split}", leave=False):
            out_name = f"{pfx}{base}_crop{i}.png"
            img.convert("RGB").save(os.path.join(split_dir, out_name))
            counts[split] += 1

    print(f"Done! Train={counts['train']} Val={counts['val']} Test={counts['test']}")
    return counts["train"], counts["val"], counts["test"]

def prepare_data(source_dir, target_dir,
                 cat_counts, crop_size=CROP_SIZE, crops_per_image=3,
                 ratios=(0.8, 0.1, 0.1), seed=SEED):
    source_root = source_dir
    for split in ("train", "val", "test"):
        clear_folder(os.path.join(target_dir, split))

    summary = {}
    for style, n in cat_counts.items():
        source = os.path.join(source_root, style)
        selected = random_select(source, n, size=crop_size, seed=seed)
        crops = get_crops(selected, crop_size=crop_size, crops_per_image=crops_per_image)
        n_train, n_val, n_test = split_data(crops, target_dir, ratios=ratios, prefix=style)
        summary[style] = (n_train, n_val, n_test)
        print(f"[{style}]: Train={n_train} Val={n_val} Test={n_test}")
    return summary

def preextract_randomcrops(source_dir, target_dir, crop_size=CROP_SIZE, crops_per_image=3):
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