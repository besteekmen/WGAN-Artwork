import os

# --- Reproducibility (set None for random seed) ---
SEED = 42

# --- Training settings ---
BATCH_SIZE = 16 # reduced from 128 to avoid OOM
# BATCH_SIZE has a major impact on how much GPU memory the code will consume!
# If not sure what batch size is appropriate:
# 1. Start at a small value
# 2. Train the model for 1 epoch
# 3. Double the batch size until errors pop up
# Ex: For MNIST, setting BATCH_SIZE to 128 is good enough, costs less than 1 GB of GPU memory.
EPOCH_NUM = 25
# EPOCH_NUM has a great impact on the training time of neural networks.
# To improve results:
# 1. Set a larger epoch number
# 2. Set a small learning rate
#lr = 2e-4 # learning rate (not used, TODO: could be a fallback)
LR_G = 2e-4 # generator learning rate
LR_D = 1e-4 # discriminator learning rate

# --- Model hyperparameters ---
IS_GATED = True # toggle between gated and standard convolutions in the fine stage
D_HIDDEN = 64 # base discriminator channels
# D_HIDDEN_LOCAL = 32 (optional, if not the same, for less params)
G_HIDDEN = 64 # base generator channels (stage)
#Z_DIM = 100 # not used!

# --- Image channels and dimensions ---
IMAGE_CHANNELS = 3 # Num of color channels
X_DIM = 64

# --- Image channels and dimensions ---
LOCAL_PATCH_SIZE = 128 # Patch size for local discriminator

# --- Loss weights (similar to Contextual Attention Yu et al 2018) ---
# TODO: do I need to change?
HOLE_LAMBDA = 6.0 # full weight for missing region (was 1.0)
VALID_LAMBDA = 1.0 # smaller for known region (was 0.1)
L1_LAMBDA = 1.0 # was 10.0
STYLE_LAMBDA = 120.0 # was 250
ADV_LAMBDA = 0.001 # small weight for adversarial loss (for stable training)
PERCEPTUAL_LAMBDA = 0.05 # If textures too blurry, try 0.1
GP_LAMBDA = 10.0 # WGAN-GP penalty weight

# --- Datasets and paths ---
OUT_PATH = 'out'
SOURCE_PATH = 'data/wikiart' # Root of dataset
DATA_PATH = 'data'
CROP_PATH = 'data/crops'
SAMPLE_PATH = 'img'
CROP_SIZE = 256
CROP_COUNT = 1 # TODO: Random crop is used to crop only 1 patch!
NUM_WORKERS = 2 # TODO: try 4
# LOG_FILE = os.path.join(OUT_PATH, 'log.txt')

# --- Labels for fake and real images (WGAN-GP: Not used anymore!) ---
#FAKE_LABEL = 0
#REAL_LABEL = 1

# --- CUDA usage ---
CUDA = True # set 'False' to train on CPU








