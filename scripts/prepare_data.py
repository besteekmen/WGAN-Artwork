import os

from config import SOURCE_PATH, DATA_PATH, CROP_SIZE, CROP_COUNT, SEED
from dataset import prepare_data

STYLE_COUNTS = {
    # TODO: Turn this dict into a file to input!
    # Chaotic marks and textured abstraction
    "Abstract_Expressionism":2300,
    "Pointillism": 512,

    # Geometric fragmentation and simplification
    "Cubism": 1800,
    "Naive_Art_Primitivism": 2000,
    "Minimalism": 485,

    # Flat color and graphic line
    "Color_Field_Painting":1200,
    "Ukiyo_e":1142,
    "Art_Nouveau_Modern":1943,

    # Saturated color and emotional hue
    "Expressionism":2500,
    "Fauvism":929,
    "Post_Impressionism":2329,

    # Fine realism and texture detail
    "Realism":2800,
    "Contemporary_Realism":473,
    "Northern_Renaissance":1012,

    # Classical form and balanced composition
    "Early_Renaissance":1233,
    "High_Renaissance":1192,
    "Mannerism_Late_Renaissance":947,
    "Rococo":913,

    # Dramatic light, mood and atmosphere
    "Baroque":1500,
    "Romanticism": 1500,
    "Impressionism": 785,
    "Symbolism":500
}

if __name__ == "__main__":
    # Extract crops (run once when the raw data is updated)
    source_dir = '../'+SOURCE_PATH
    target_dir = '../'+DATA_PATH
    crop_size = CROP_SIZE
    crop_count = CROP_COUNT
    ratios = (0.8, 0.1, 0.1)  # Split ratio for train/val/test sets
    seed = SEED

    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    summary = prepare_data(
        source_dir=source_dir,
        target_dir=target_dir,
        cat_counts=STYLE_COUNTS,
        crop_size=crop_size,
        crops_per_image=crop_count,
        ratios=ratios,
        seed=seed
    )

    print(f"Summary per style:")
    for style, (tr, va, te) in summary.items():
        print(f"{style:28s}: train={tr:5d}, val={va:5d}, test={te:5d}")



