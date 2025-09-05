import os

# -------------------------------------------
# Reproducibility (set None for random seed)
# -------------------------------------------
SEED = 42

# -------------------------------------------
# Training settings
# -------------------------------------------
BATCH_SIZE = 16 # reduced from 128 to avoid OOM
# BATCH_SIZE has a major impact on how much GPU memory the code will consume!
# If not sure what batch size is appropriate:
# 1. Start at a small value
# 2. Train the model for 1 epoch
# 3. Double the batch size until errors pop up
# Ex: For MNIST, setting BATCH_SIZE to 128 is good enough, costs less than 1 GB of GPU memory.
EPOCH_NUM = 40 # Ideally 50 to 100!
# EPOCH_NUM has a great impact on the training time of neural networks.
# To improve results:
# 1. Set a larger epoch number
# 2. Set a small learning rate
#lr = 2e-4 # learning rate (not used, TODO: could be a fallback)
LR_G = 8e-5 # generator learning rate, change to 1e-4 if NaN g loss
LR_D = 4e-5 # discriminator learning rate

# -------------------------------------------
# Model hyperparameters
# -------------------------------------------
IS_GATED = True # toggle between gated and standard convolutions in the fine stage
D_HIDDEN = 64 # base discriminator channels
# D_HIDDEN_LOCAL = 32 (optional, if not the same, for less params)
G_HIDDEN = 64 # base generator channels (stage)
#Z_DIM = 100 # not used!

# -------------------------------------------
# Image channels and dimensions
# -------------------------------------------
IMAGE_CHANNELS = 3 # Num of color channels
X_DIM = 64
LOCAL_PATCH_SIZE = 128 # Patch size for local discriminator

# -------------------------------------------
# Loss weights w/wo schedule
# Epochs 1-10: Model learns what to fill in (Strong HOLE/L1, weak STYLE/ADV)
# Epochs 11-20: Model learns how it should look (STYLE/PERCEPTUAL starts to effect)
# Epochs 21-40: Model sharpens the textures and realism w/o giving up stability (ADV increases)
# TODO: Switch to smooth interpolation if you train longer (80 - 100 epochs)
# -------------------------------------------
LOSS_SCHEDULE = {
    "HOLE_LAMBDA": [(1, 20), (10, 12), (20, 8), (40, 6)], # Reconstruct missing regions strongly early and learn structure
    "VALID_LAMBDA": [(1, 1.0), (40, 1.0)], # Keep small and stable (do not penalize know too much)
    "L1_LAMBDA": [(1, 15), (10, 8), (20, 4), (40, 2)], # Pixel-wise loss (drop gradually to reduce blur)
    "STYLE_LAMBDA": [(1, 0.1), (5, 5), (15, 20), (30, 40), (40, 60)], # Gradual increase to focus on details after building textures!
    "ADV_LAMBDA": [(1, 0.0), (10, 0.01), (20, 0.05), (30, 0.08), (40, 0.15)], # Increase gradually to avoid instability and artifacts
    "PERCEPTUAL_LAMBDA": [(1, 0.0), (5, 0.05), (15, 0.1), (25, 0.2), (40, 0.4)], # Increase mid-training for details, but do not dominate!
}

SCALE_SCHEDULE = {
    "inter": [(1, 0.5), (10, 0.3), (20, 0.2), (40, 0.1)],
    "fine": [(1, 0.5), (10, 0.7), (20, 0.8), (40, 0.9)],
} # multiscale factor weights with schedule to start stable and focus on details later

GP_LAMBDA = 10.0 # WGAN-GP penalty weight
GP_SCHEDULE = [(1,1), (10, 2), (20,4), (40,8)]
GP_SUBSET = 6 # number of samples per batch to compute GP

# -------------------------------------------
# Datasets and paths
# -------------------------------------------
OUT_PATH = 'out'
SOURCE_PATH = 'data/wikiart' # Root of dataset
DATA_PATH = 'data'
CROP_PATH = 'data/crops'
SAMPLE_PATH = 'img'
CROP_SIZE = 256
CROP_COUNT = 1 # TODO: Random crop is used to crop only 1 patch!
NUM_WORKERS = 8 # TODO: try 12?
SAVE_FREQ = 500
# LOG_FILE = os.path.join(OUT_PATH, 'log.txt')

# -------------------------------------------
# CUDA usage
# -------------------------------------------
CUDA = True # set 'False' to train on CPU








