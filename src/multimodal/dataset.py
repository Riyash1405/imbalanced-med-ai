"""
A flexible dataset supporting:
- CT image arrays (numpy) or image file paths
- genomic feature arrays
- clinical tabular features (optional)
This dataset demonstrates loading and transforms for training.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MultiModalDataset(Dataset):
    def __init__(self, img_paths=None, img_arrays=None, gene_arrays=None, clinical=None, labels=None, img_transform=None):
        # Provide either img_paths or img_arrays
        self.img_paths = img_paths
        self.img_arrays = img_arrays
        self.gene = gene_arrays
        self.clinical = clinical
        self.labels = labels
        self.transform = img_transform

        assert (img_paths is not None) or (img_arrays is not None), "Provide images"
        assert self.gene is not None, "Provide gene arrays"
        assert self.labels is not None, "Provide labels"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.img_arrays is not None:
            img = self.img_arrays[idx]
        else:
            img = np.array(Image.open(self.img_paths[idx]).convert("L"))
        img = img.astype("float32")
        if self.transform:
            img = self.transform(img)
        else:
            # normalize and add channel
            img = (img - img.mean()) / (img.std()+1e-8)
            img = np.expand_dims(img, 0)
        gene = self.gene[idx].astype("float32")
        label = np.float32(self.labels[idx])
        return torch.tensor(img), torch.tensor(gene), torch.tensor(label)
