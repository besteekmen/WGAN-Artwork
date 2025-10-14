# ALL ADJUSTABLE HYPERPARAMS
# TODO: rather than a config file, try parsing later as arguments

# --- Reproducibility (set None for random seed) ---
SEED = 42
VAL_SEED = 1234

# --- Global helpers ---
EPS = 1e-8 # Epsilon for safe mathematical operations (like division)
TOL = 7 # Tolerance for early stopping (set higher than 50 to omit)
SAVE_FREQ = 200 # If epoch time is high, increase to save visuals less often
CHECKPOINT_EVERY = 1 # Frequency of model saving

# --- Training settings ---
BATCH_SIZE = 16 # reduced from 128 to avoid OOM on local laptop
EPOCH_NUM = 50
LR_G = 1e-4 # generator learning rate, change to 1e-4 if NaN g loss
LR_D = 2e-4 # discriminator learning rate, was also 1e-4, reduced for G to sharpen details
OPTIM_BETAS = (0.0, 0.9)

# --- Model hyperparameters ---
D_HIDDEN = 64 # base discriminator channels
G_HIDDEN = 64 # base generator channels

# --- Image channels and dimensions ---
IMAGE_CHANNELS = 3 # Num of color channels (RGB)
LOCAL_PATCH_SIZE = 128 # Patch size for local discriminator
# Local patch size could be adapted, but keep it as it is!
JITTER = 16 # to remove boundary artefacts or local patch crop

# --- Mask settings ---
IRR_RATIO_SCHEDULE = [ # irregular masks ratio was constant at 0.3 before
    (0, 0.20),
    (20, 0.30),
    (35, 0.50)
]

# --- Loss weights and hyperparameters ---
HOLE_LAMBDA = 4.0 # full weight for missing region, reduced from 6.0 to avoid large gradients
VALID_LAMBDA = 1.0 # smaller for known region (was 0.1)
L1_LAMBDA = 1.0 # was 10.0 reconstruction loss weight
TV_LAMBDA = 1e-4
FM_LAMBDA_SCHEDULE = [
    (0, 0.0),
    (5, 0.5),
    (10, 1.0),
    (20, 2.0)
]
EDGE_LAMBDA_SCHEDULE = [ # was constant before as EDGE_LAMBDA = 0.05
    (0, 0.005),
    (10, 0.015),
    (20, 0.024),
    (30, 0.030)
]
STYLE_LAMBDA_SCHEDULE = [ # was constant before as STYLE_LAMBDA = 60.0
    # Too strong style cues at the beginning harmed stabilization of shapes
    (0, 18.0),
    (10, 36.0),
    (20, 48.0),
    (30, 60.0)
]
ADV_LAMBDA_SCHEDULE = [ # was constant before as ADV_LAMBDA = 0.005
    (0, 0.005),
    (30, 0.0075),
    (40, 0.01)
]
PERCEPTUAL_LAMBDA_SCHEDULE = [ # was constant before as PERCEPTUAL_LAMBDA = 0.1
    (0, 0.1),
    (25, 0.075),
    (35, 0.05)
]
DIF_RING_SCHEDULE = [ # no need for schedule for now
    (0, 2)
    #(5, 1)
    #(20, 1)
]
DIF_SCALE_SCHEDULE = [ # one scalar to rule them all :)
    (0, 1.00),
    (5, 0.85),
    (15, 0.65),
    (25, 0.50)
]
ADV_LAMBDA = 0.005 # small weight for adversarial loss (for stable training)
PERCEPTUAL_LAMBDA = 0.1
# If textures too blurry, try 0.1
GP_LAMBDA = 20.0 # WGAN-GP penalty weight
SCALES = [1.0, 0.5, 0.25] # multiscale factors
TV_RING = 2

# --- Datasets and paths ---
# TODO: set as arguments to parse later
OUT_PATH = 'out'
SOURCE_PATH = 'data/original' # Root of dataset
DATA_PATH = 'data'
CROP_PATH = 'data/crops'
SAMPLE_PATH = 'img'
CROP_SIZE = 256
CROP_COUNT = 1 # TODO: Random crop is used to crop only 1 patch!
NUM_WORKERS = 4
LOG_PATH = 'logs'
CHECK_PATH = 'checkpoints'
TRAIN_LOG_FILE = 'train.log'
LOAD_MODEL = False

# --- CUDA usage ---
CUDA = True # set 'False' to train on CPU, but dont!
