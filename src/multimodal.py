# src/multimodal.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from .config import DEVICE

# Simple dataset to pair image files with genomics vector and label
class MultimodalDataset(Dataset):
    def __init__(self, df_meta, img_root, img_col='patient_id', img_exts=('png','jpg','jpeg'), transform=None):
        """
        df_meta: pandas DataFrame that contains columns [img_id_column, genomics features..., label]
        img_root: folder containing images named <patient_id>.<ext>
        """
        self.df = df_meta.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.img_col = img_col
        # Identify genomics columns as those that are not img_col or label
        self.label_col = self.df.columns[-1]
        self.genomic_cols = [c for c in self.df.columns if c not in [self.img_col, self.label_col]]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id = str(row[self.img_col])
        # find file
        found = None
        for ext in ('png','jpg','jpeg'):
            p = os.path.join(self.img_root, f"{img_id}.{ext}")
            if os.path.exists(p):
                found = p
                break
        if found is None:
            # return zeros if image not found
            img = Image.new('RGB', (224,224), (0,0,0))
        else:
            img = Image.open(found).convert('RGB')
        img = self.transform(img)
        genomics = torch.tensor(row[self.genomic_cols].astype(float).values, dtype=torch.float32)
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return img, genomics, label

# Encoders and fusion model
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        # remove fc
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, out_dim)

    def forward(self, x):
        x = self.features(x)          # [B, C, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        return x

class GenomicsEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, max(128, input_dim//2)),
            nn.ReLU(),
            nn.BatchNorm1d(max(128, input_dim//2)),
            nn.Linear(max(128, input_dim//2), out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class FusionClassifier(nn.Module):
    def __init__(self, img_dim=128, gen_dim=64, hidden=128, n_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + gen_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, img_feat, gen_feat):
        x = torch.cat([img_feat, gen_feat], dim=1)
        return self.fc(x)

# simple training loop
def train_fusion(df_meta, img_root, outdir, epochs=8, batch_size=16, lr=1e-3, device=DEVICE):
    if not os.path.exists(img_root):
        print("Image folder not found:", img_root)
        return False
    ds = MultimodalDataset(df_meta, img_root)
    if len(ds) == 0:
        print("Empty multimodal dataset")
        return False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # instantiate models
    img_enc = ImageEncoder(out_dim=128).to(device)
    gen_enc = GenomicsEncoder(input_dim=len(ds.genomic_cols), out_dim=64).to(device)
    fusion = FusionClassifier(img_dim=128, gen_dim=64, hidden=128, n_classes=2).to(device)

    params = list(img_enc.parameters()) + list(gen_enc.parameters()) + list(fusion.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        img_enc.train(); gen_enc.train(); fusion.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for imgs, gens, labels in loader:
            imgs = imgs.to(device); gens = gens.to(device); labels = labels.to(device)
            opt.zero_grad()
            img_feats = img_enc(imgs)
            gen_feats = gen_enc(gens)
            logits = fusion(img_feats, gen_feats)
            loss = criterion(logits, labels)
            loss.backward(); opt.step()
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/total:.4f} acc={correct/total:.4f}")

    # Save model states
    torch.save({
        'img_enc': img_enc.state_dict(),
        'gen_enc': gen_enc.state_dict(),
        'fusion': fusion.state_dict()
    }, os.path.join(outdir, "multimodal_model.pt"))
    print("Saved multimodal model to", outdir)
    return True
