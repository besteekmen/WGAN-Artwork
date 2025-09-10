import errno
import os
import shutil
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from config import *

# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------
def is_cuda():
    """Check cuda based on settings and availability."""
    return CUDA and torch.cuda.is_available()

def half_precision():
    """Use autocast with half precision."""
    return amp.autocast(device_type='cuda', dtype=torch.float16)

def full_precision():
    """Use full precision."""
    return amp.autocast(device_type='cuda', dtype=torch.float16, enabled=False)

def get_device():
    """Get the current device based on settings and availability."""
    # will send this as parameter to all functions using device
    # like func(device=get_device()): ss.to(device)
    return torch.device(
        "cuda:0" if is_cuda() else "cpu")

def print_device():
    """Print the current device name and version if available."""
    print(f"Device: {get_device()}")
    if is_cuda():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU-only")

def set_seed(seed=SEED):
    """Set seed for reproducibility."""
    seed_val = np.random.randint(1, 10000) if seed is None else seed
    print(f"Random Seed: {seed_val}")
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if is_cuda():
        torch.cuda.manual_seed(seed_val)

def to_np(var):
    """Export torch.Tensor to NumPy array."""
    return var.detach().cpu().numpy()

def to_unit(tensor: torch.Tensor) -> torch.Tensor:
    """Rescale tensor from [-1,1] to [0,1]."""
    return (tensor + 1) / 2

def to_signed(tensor: torch.Tensor) -> torch.Tensor:
    """Rescale tensor from [0,1] to [-1,1]."""
    return (tensor * 2) - 1

def create_folder(folder_path):
    """Create a folder if it does not exist."""
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise

def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidentally deleted.
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)

def make_run_directory(out_path=OUT_PATH, log_path=LOG_PATH, check_path=CHECK_PATH):
    """Make the output directory for the current training run."""
    clear_folder(out_path)
    log = os.path.join(out_path, log_path) # training logs
    check = os.path.join(out_path, check_path) # model checkpoints
    create_folder(log)
    create_folder(check)
    return out_path, log, check

class StdOut(object):
    """Redirect all messages from stdout to the log file,
    and print to console as well.
    """
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()