from dataset import CroppedImageDataset
# from train import *

def main():
    crops_dir = "data/crops"
    dataset = CroppedImageDataset(crops_dir)
    # train_model(dataset)

if __name__ == "__main__":
    main()