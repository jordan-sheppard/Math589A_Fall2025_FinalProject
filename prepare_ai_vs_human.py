# make_project_data.py
import os
import random
import numpy as np
from PIL import Image

try:
    from scipy.io import savemat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DATA_ROOT = "data"          # contains "human" and "ai" subdirs
IMG_SIZE = 64               # final size: IMG_SIZE x IMG_SIZE
TRAIN_FRACTION = 0.7
MAX_PER_CLASS = 400         # you can lower/raise this

def load_images_from_folder(folder, label, max_count=None):
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    random.shuffle(paths)
    if max_count is not None:
        paths = paths[:max_count]

    images = []
    labels = []
    for path in paths:
        img = Image.open(path).convert("L")         # grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE))      # downsample
        arr = np.array(img, dtype=np.float32)       # shape (H, W)
        images.append(arr)
        labels.append(label)
    return images, labels

def main():
    random.seed(0)

    human_dir = os.path.join(DATA_ROOT, "human")
    ai_dir    = os.path.join(DATA_ROOT, "ai")

    human_imgs, human_labels = load_images_from_folder(
        human_dir, label=0, max_count=MAX_PER_CLASS
    )
    ai_imgs, ai_labels = load_images_from_folder(
        ai_dir, label=1, max_count=MAX_PER_CLASS
    )

    X = np.array(human_imgs + ai_imgs)   # (N, H, W)
    y = np.array(human_labels + ai_labels, dtype=np.int64)

    # Shuffle
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Optional: scale to [0,1]
    X = X / 255.0

    # Train/test split
    N = len(y)
    N_train = int(TRAIN_FRACTION * N)

    X_train = X[:N_train]
    y_train = y[:N_train]
    X_test  = X[N_train:]
    y_test  = y[N_train:]

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # Save NPZ for the Python starter
    np.savez_compressed(
        "project_data.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Optionally also save a MAT file for MATLAB starter
    if HAS_SCIPY:
        savemat(
            "project_data.mat",
            {
                "X_train": X_train,
                "y_train": y_train.reshape(-1, 1),
                "X_test": X_test,
                "y_test": y_test.reshape(-1, 1),
            },
        )
        print("Saved project_data.mat")
    else:
        print("scipy not installed; only project_data.npz saved.")

if __name__ == "__main__":
    main()
