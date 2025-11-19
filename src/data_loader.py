import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, RANDOM_STATE
from pathlib import Path

def load_tabular(dataset="heart"):
    if dataset == "heart":
        p = DATA_DIR / "heart.csv"
        df = pd.read_csv(p)
        # standard expected column name 'target'
        if "target" not in df.columns:
            raise ValueError("heart.csv must have 'target' column")
    elif dataset == "parkinsons":
        p = DATA_DIR / "parkinsons.csv"
        df = pd.read_csv(p)
        if "status" in df.columns:
            df = df.rename(columns={"status": "target"})
        if "target" not in df.columns:
            raise ValueError("parkinsons.csv must have 'status' column")
    else:
        raise ValueError("unknown dataset")
    # basic cleaning: drop constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=const_cols)
    return df

def train_test_split_tabular(df, test_size=0.2, stratify_col="target"):
    X = df.drop(columns=[stratify_col])
    y = df[stratify_col]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

# Placeholder CT loader (expects folder of png/jpg)
from PIL import Image
def load_ct_images(path, limit=None, img_size=(128,128), normalize=True):
    path = Path(path)
    files = list(path.glob("**/*.png")) + list(path.glob("**/*.jpg")) + list(path.glob("**/*.jpeg"))
    if limit:
        files = files[:limit]
    imgs = []
    for f in files:
        im = Image.open(f).convert("L").resize(img_size)
        arr = np.array(im, dtype=np.float32)
        if normalize:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        imgs.append(arr)
    return np.stack(imgs, axis=0)
