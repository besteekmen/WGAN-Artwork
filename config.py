# --- Reproducibility (set None for random seed) ---
SEED = 42
VAL_SEED = 1234

# --- Global helpers ---
EPS = 1e-8 # Epsilon for safe mathematical operations
TOL = 5 # Tolerance for early stopping
SAVE_FREQ = 200
CHECKPOINT_EVERY = 1 # Frequency of model saving

# --- Training settings ---
BATCH_SIZE = 16 # reduced from 128 to avoid OOM
# BATCH_SIZE has a major impact on how much GPU memory the code will consume!
# If not sure what batch size is appropriate:
# 1. Start at a small value
# 2. Train the model for 1 epoch
# 3. Double the batch size until errors pop up
# Ex: For MNIST, setting BATCH_SIZE to 128 is good enough, costs less than 1 GB of GPU memory.
EPOCH_NUM = 50
# EPOCH_NUM has a great impact on the training time of neural networks.
# To improve results:
# 1. Set a larger epoch number
# 2. Set a small learning rate
#lr = 2e-4 # learning rate (not used, TODO: could be a fallback)
# TODO: If good texture but broken structure, prioritize global (Izuka et al. 2017)
# TODO: If discriminators are weak (not decreasing loss), use update ratios (WGAN-GP)
# If discriminator gets perfect quickly, generator gradients may vanish!
# Hence make learning slower for D and faster for G (Contextual Attention GAN, Yu et al. 2018)
LR_G = 1e-4 # generator learning rate, change to 1e-4 if NaN g loss
LR_D = 8e-5 # discriminator learning rate, was also 1e-4, reduced for G to sharpen details
OPTIM_BETAS = (0.0, 0.9)

# --- Model hyperparameters ---
IS_GATED = True # toggle between gated and standard convolutions in the fine stage
D_HIDDEN = 64 # base discriminator channels
# D_HIDDEN_LOCAL = 32 (optional, if not the same, for less params)
G_HIDDEN = 64 # base generator channels (stage)

# --- Image channels and dimensions ---
IMAGE_CHANNELS = 3 # Num of color channels
X_DIM = 64
LOCAL_PATCH_SIZE = 128 # Patch size for local discriminator
JITTER = 8 #16

# --- Mask settings ---
IRR_RATIO_SCHEDULE = [ # irregular masks ratio was constant at 0.3 before
    (0, 0.20),
    (20, 0.30),
    (35, 0.50)
]

# --- Loss weights and hyperparameters ---
HOLE_LAMBDA = 7.5 # 4.0 # full weight for missing region, reduced from 6.0 to avoid large gradients
VALID_LAMBDA = 0.1 # 0.2 # 1.0 # smaller for known region (was 0.1)
L1_LAMBDA = 0.25 #0.5 #1.0 # was 10.0 reconstruction loss weight
LPIPS_LAMBDA = 0.15 # 0.05 or 0.15 later
TV_LAMBDA = 2e-4
EDGE_LAMBDA_SCHEDULE = [ # was constant before as EDGE_LAMBDA = 0.05
    (0, 0.07),
    (10, 0.10),
    (20, 0.12)
]
VAL_EDGE_LAMBDA = 0.10
STYLE_LAMBDA_SCHEDULE = [ # was constant before as STYLE_LAMBDA = 60.0
    (0, 384.0), # 24 x 16
    (5, 576.0), # 36 x 16
    (10, 704.0), # 44 x 16
    (20, 832.0) # 52 x 16
]
VAL_STYLE_LAMBDA = 704 #60.0
ADV_LAMBDA_SCHEDULE = [ # was constant before as ADV_LAMBDA = 0.005
    (0, 0.006),
    (3, 0.0085),
    (10, 0.014)
]
PERCEPTUAL_LAMBDA_SCHEDULE = [ # was constant before as PERCEPTUAL_LAMBDA = 0.1
    (0, 0.03),
    (5, 0.05),
    (15, 0.06)
]
VAL_PERCEPTUAL_LAMBDA = 0.05
ADV_LAMBDA = 0.005 # small weight for adversarial loss (for stable training)
PERCEPTUAL_LAMBDA = 0.1 # Reduced from 0.05 for smoother early training
# If textures too blurry, try 0.1
GP_LAMBDA = 10.0 # WGAN-GP penalty weight
SCALES = [1.0, 0.5, 0.25] # multiscale factors
SCALE_WEIGHTS = [1.0, 0.5, 0.25] # multiscale factor weights
EDGE_RING = 2
LPIPS_RING = 3
VGG_RING = 3

# --- Datasets and paths ---
OUT_PATH = 'out'
SOURCE_PATH = 'data/wikiart' # Root of dataset
DATA_PATH = 'data'
CROP_PATH = 'data/crops'
SAMPLE_PATH = 'img'
CROP_SIZE = 256
CROP_COUNT = 1 # TODO: Random crop is used to crop only 1 patch!
NUM_WORKERS = 4 # TODO: try 4
LOG_PATH = 'logs'
CHECK_PATH = 'checkpoints'
TRAIN_LOG_FILE = 'train.log'
LOAD_MODEL = False

# --- CUDA usage ---
CUDA = True # set 'False' to train on CPU








