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
LR_G = 1e-4 #8e-5 # generator learning rate, change to 1e-4 if NaN g loss
LR_D = 5e-5 #4e-5 # discriminator learning rate

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
# Epochs 1-5: Model learns what to fill in (Strong HOLE/L1, weak STYLE/ADV)
# Epochs 5-15: Model learns how it should look (STYLE/PERCEPTUAL/ADV starts to effect)
# Epochs 20-40: Model sharpens the textures and realism w/o giving up stability (ADV increases)
# TODO: Switch to smooth interpolation to train longer (80 - 100 epochs)
# -------------------------------------------
LOSS_SCHEDULE = {
    #"HOLE_LAMBDA": [(1, 20), (10, 12), (20, 8), (40, 6)], # Reconstruct missing regions strongly early and learn structure
    "HOLE_LAMBDA": [(1, 20), (5, 12), (15, 6), (30, 3)], # Reconstruct missing regions strongly early and learn structure
    "VALID_LAMBDA": [(1, 1.0), (40, 1.0)], # (50, 1.0) # Keep small and stable (do not penalize know too much)
    #"L1_LAMBDA": [(1, 15), (10, 8), (20, 4), (40, 2)], # Pixel-wise loss (drop gradually to reduce blur)
    "L1_LAMBDA": [(1, 10), (5, 5), (15, 2), (30, 1)], # Pixel-wise loss (drop gradually to reduce blur)
    "GLOBAL_LAMBDA":[(1, 1.0), (40, 1.0)], # (50, 1.0) # Keep stable
    "LOCAL_LAMBDA": [(1, 1.0), (10, 1.2), (20, 1.5)], # Emphasize local details later
    #"STYLE_LAMBDA": [(1, 0.1), (5, 5), (15, 20), (30, 40), (40, 60)], # Gradual increase to focus on details after building textures!
    "STYLE_LAMBDA": [(1, 1.0), (5, 5), (15, 10), (30, 20), (40, 30)], # (40, 60) -> (50, 1.0) # Gradual increase to focus on details after building textures!
    #"ADV_LAMBDA": [(1, 0.0), (10, 0.01), (20, 0.05), (30, 0.08), (40, 0.15)], # Increase gradually to avoid instability and artifacts
    "ADV_LAMBDA": [(1, 0.02), (5, 0.05), (15, 0.1), (30, 0.2), (40, 0.3)], # (50, 0.1) # Increase gradually to avoid instability and artifacts
    #"PERCEPTUAL_LAMBDA": [(1, 0.0), (5, 0.05), (15, 0.1), (25, 0.2), (40, 0.4)] # Increase mid-training for details, but do not dominate!
    "PERCEPTUAL_LAMBDA": [(1, 0.0), (5, 0.1), (15, 0.2), (30, 0.25), (40, 0.3)] # (50, 0.5) # Increase mid-training for details, but do not dominate!
}

SCALE_SCHEDULE = {
    "inter": [(1, 0.6), (10, 0.45), (20, 0.35), (40, 0.25)], # (50, 0.15)
    "fine": [(1, 0.4), (10, 0.55), (20, 0.65), (40, 0.75)], # (50, 0.5)
} # multiscale factor weights with schedule to start stable and focus on details later

GP_LAMBDA = 10.0 # WGAN-GP penalty weight
#GP_SCHEDULE = [(1,1), (10, 2), (20,4), (40,8)]
GP_SCHEDULE = [(1,2), (5, 3), (15,4), (30,5), (40,8)] # (50, 8)
GP_SUBSET = 6 # number of samples per batch to compute GP

# -------------------------------------------
# Datasets and paths
# -------------------------------------------
OUT_PATH = 'out'
EVAL_PATH = 'eval_out'
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








