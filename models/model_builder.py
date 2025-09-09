import os
import torch
from torch import optim
from models.generator import Generator
from models.discriminator import GlobalDiscriminator, LocalDiscriminator, Discriminator
from models.weights_init import weights_init_normal
from utils.utils import get_device
from config import *

def init_nets(device=get_device()):
    """Initialize the networks on the current device."""
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

def init_model(netG, globalD, localD, optimG, optimGD, optimLD):
    """Initialize the model with new weights."""
    netG.apply(weights_init_normal)
    netG.upsample_init()  # used to avoid initial patchy results
    # print(netG) # DEBUG only: causes repetitive printing
    globalD.apply(weights_init_normal)
    localD.apply(weights_init_normal)
    # print(globalD) # DEBUG only

def save_checkpoint(epoch, netG, globalD, localD, optimG, optimGD, optimLD, path):
    """Save the checkpoint."""
    checkpoint = {
        'netG': netG.state_dict(),
        'globalD': globalD.state_dict(),
        'localD': localD.state_dict(),
        'optimG': optimG.state_dict(),
        'optimGD': optimGD.state_dict(),
        'optimLD': optimLD.state_dict()
    }
    torch.save(checkpoint, os.path.join(path, f'checkpoint{epoch + 1}.pth.tar'))

def load_checkpoint(netG, globalD, localD, optimG, optimGD, optimLD, checkpoint_file):
    """Load the checkpoint."""
    print(f'Loading checkpoint from {checkpoint_file}...')
    checkpoint = torch.load(checkpoint_file)  # i.e. "checkpoint5.pth.tar"
    netG.load_state_dict(checkpoint['netG'])
    globalD.load_state_dict(checkpoint['globalD'])
    localD.load_state_dict(checkpoint['localD'])
    # Use previous state weights
    optimG.load_state_dict(checkpoint['optimG'])
    optimGD.load_state_dict(checkpoint['optimGD'])
    optimLD.load_state_dict(checkpoint['optimLD'])