import os

# --- Training settings ---
BATCH_SIZE = 128
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
lr = 2e-4 # learning rate

# --- Model hyperparameters ---
D_HIDDEN = 64
G_HIDDEN = 64
Z_DIM = 100

# --- Image channels and dimensions ---
IMAGE_CHANNEL = 1 # Num of color channels
X_DIM = 64

# --- Datasets and paths ---
SOURCE_PATH = 'data/wikiart' # Root of dataset
CROP_PATH = 'data/crops'
SAMPLE_PATH = 'img'
CROP_SIZE = 256
CROP_COUNT = 1
# LOG_FILE = os.path.join(OUT_PATH, 'log.txt')

# --- Labels for fake and real images ---
FAKE_LABEL = 0
REAL_LABEL = 1

# --- CUDA usage ---
CUDA = True # set 'False' to train on CPU

# --- Seed (set None for random seed) ---
seed = 1








