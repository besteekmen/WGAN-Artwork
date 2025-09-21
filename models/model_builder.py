import os
import re
import torch
from torch import optim
from models.generator import Generator
from models.discriminator import GlobalDiscriminator, LocalDiscriminator
from models.weights_init import weights_init_normal, bias_init_gate
from utils.utils import get_device
from config import *

def init_nets(device=None):
    """Initialize the networks on the current device."""
    if device is None:
        device = get_device()
    netG = Generator().to(device)
    globalD = GlobalDiscriminator().to(device)
    localD = LocalDiscriminator().to(device)
    return netG, globalD, localD

def init_optimizers(netG, globalD, localD):
    """Initialize the optimizers for model networks."""
    optimG = optim.Adam(netG.parameters(), lr=LR_G, betas=OPTIM_BETAS)
    optimGD = optim.Adam(globalD.parameters(), lr=LR_D, betas=OPTIM_BETAS)
    optimLD = optim.Adam(localD.parameters(), lr=LR_D, betas=OPTIM_BETAS)
    return optimG, optimGD, optimLD

def setup_model(netG, globalD, localD, optimG, optimGD, optimLD, check_path, device=None, load_file="", is_pretrained=LOAD_MODEL):
    """Setup the model either by initializing or loading."""
    if is_pretrained:
        start_epoch = load_model(netG, globalD, localD,
                                 optimG, optimGD, optimLD,
                                 check_path, load_file, device) # add checkpoint file name as the last parameter
    else:
        start_epoch = init_model(netG, globalD, localD, optimG, optimGD, optimLD) # w/o checkpoint!
    return start_epoch

def init_model(netG, globalD, localD, optimG, optimGD, optimLD):
    """Initialize the model with new weights."""
    netG.apply(weights_init_normal)
    netG.upsample_init()  # used to avoid initial patchy results
    netG.apply(bias_init_gate) # start gates open for better detail flow
    # print(netG) # DEBUG only: causes repetitive printing
    globalD.apply(weights_init_normal)
    localD.apply(weights_init_normal)
    # print(globalD) # DEBUG only
    start_epoch = 0
    return start_epoch

def load_model(netG, globalD, localD, optimG, optimGD, optimLD, check_path, load_file, device=None):
    """Load the model from the checkpoint in given location."""
    start_epoch = 0
    checkpoint_file = os.path.join(check_path, load_file)
    load_epoch = load_checkpoint(netG, globalD, localD, optimG, optimGD, optimLD, checkpoint_file, device)
    if load_epoch is not None:
        start_epoch = load_epoch + 1
    else:  # for older checkpoints without epoch number
        ep = re.search(r'checkpoint(\d+)\.pth\.tar$', os.path.basename(checkpoint_file))
        start_epoch = int(ep.group(1)) if ep is not None else 0
    # if a checkpoint loaded and learning rate changed, use below
    # for pg in optimGD.param_groups: pg['lr'] = LR_D
    # for pg in optimLD.param_groups: pg['lr'] = LR_D
    # for pg in optimG.param_groups: pg['lr'] = LR_G
    return start_epoch

def save_checkpoint(epoch, netG, globalD, localD, optimG, optimGD, optimLD, path):
    """Save the checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'netG': netG.state_dict(),
        'globalD': globalD.state_dict(),
        'localD': localD.state_dict(),
        'optimG': optimG.state_dict(),
        'optimGD': optimGD.state_dict(),
        'optimLD': optimLD.state_dict()
    }
    torch.save(checkpoint, os.path.join(path, f'checkpoint{epoch + 1}.pth.tar'))

def load_checkpoint(netG, globalD, localD, optimG, optimGD, optimLD, checkpoint_file, device):
    """Load the checkpoint."""
    if device is None:
        device = get_device()
    print(f'Loading checkpoint from {checkpoint_file}...')
    checkpoint = torch.load(checkpoint_file, map_location=device)  # i.e. "checkpoint5.pth.tar"
    netG.load_state_dict(checkpoint['netG'])
    globalD.load_state_dict(checkpoint['globalD'])
    localD.load_state_dict(checkpoint['localD'])
    # Use previous state weights
    optimG.load_state_dict(checkpoint['optimG'])
    optimGD.load_state_dict(checkpoint['optimGD'])
    optimLD.load_state_dict(checkpoint['optimLD'])
    return checkpoint.get('epoch', None)

def forward_pass(netG, image, mask_hole):
    """Perform forward pass and return generated and composite images."""
    fake = netG(image, mask_hole)
    composite = fake * mask_hole + image * (1.0 - mask_hole)
    return fake, composite