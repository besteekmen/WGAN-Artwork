import os
import shutil
import matplotlib.pyplot as plt

from PIL import Image
from config import SOURCE_PATH, CROP_PATH, CROP_SIZE, CROP_COUNT, SAMPLE_PATH
from utils import clear_folder
from data_utils import preextract_fivecrops, preextract_randomcrops, CroppedImageDataset

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
    target_dir = '../'+CROP_PATH
    save_dir = '../'+SAMPLE_PATH
    #preextract_fivecrops(source_dir, target_dir, CROP_SIZE)
    #preextract_randomcrops(source_dir, target_dir, CROP_SIZE, CROP_COUNT)

    # Load dataset and show some samples
    dataset = CroppedImageDataset(crops_dir=target_dir)
    display_samples(dataset, num_samples=10, save=True, save_dir=save_dir)



