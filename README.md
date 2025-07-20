# WGAN-Artwork Project

This project implements a **Wasserstein GAN (WGAN)** using **PyTorch** to inpaint artwork images.

---

## 📁 Project Structure

```
wgan_artwork/
│
├── config.py               # Configuration variables (device, hyperparameters, paths)
├── utils.py                # General-purpose utility functions (e.g., seeding, logging)
├── train.py                # Main training script
├── evaluate.py             # Image generation and visualization after training
│
├── models/
│   ├── generator.py        # Generator model definition
│   ├── discriminator.py    # Discriminator model definition
│   └── weights_init.py     # Custom weights initialization function
│
├── requirements.txt        # Project dependencies
└── README.md               # You are here
```

---

## ⚙️ Setup

### 1. Create and activate a virtual environment (optional but recommended):

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 📌 Train the Model:

```bash
python train.py
```

This will start training using the parameters defined in `config.py`. Model checkpoints will be saved periodically in the `output/` folder.

### 🎨 Generate Samples:

After training, generate images using the saved Generator:

```bash
python evaluate.py
```

---

## ⚙️ Configuration

Adjust the following settings in `config.py`:

- Batch size
- Learning rates
- Latent dimension (`Z_DIM`)
- Device (CPU vs CUDA)
- Output directories
- Model save/load paths

---

## 🧩 Requirements

Main dependencies:

- Python 3.8+
- torch
- torchvision
- numpy
- tqdm
- matplotlib

Install them all with:

```bash
pip install -r requirements.txt
```

---

## 📸 Sample Output

The loss curves progressed as below:

![Loss Curve](/img/loss_curve.png)

Generated fake samples at epoch 24 and batch 400:

![Fake samples](/img/fake_samples_epoch24_batch400.png)
---

## 📜 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 🙏 Acknowledgements

- Based on the original DCGAN paper: *Radford et al., 2015*.
- PyTorch community tutorials and best practices.
