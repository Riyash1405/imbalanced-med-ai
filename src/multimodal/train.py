"""
Training loop for the multimodal model with:
- TensorBoard logging
- Early stopping
- LR scheduler
- Model checkpoint saving
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from .models import CTEncoder, GeneTransformer, FusionClassifier
from .dataset import MultiModalDataset
from ..config import MODELS_DIR, DEVICE, RANDOM_STATE
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalNet(nn.Module):
    def __init__(self, img_channels=1, gene_dim=500, img_size=128):
        super().__init__()
        self.ct = CTEncoder(in_channels=img_channels, base_filters=16, img_size=img_size)
        self.gene = GeneTransformer(input_dim=gene_dim, d_model=256, nhead=8, num_layers=2)
        self.classifier = FusionClassifier(ct_out_dim=self.ct.out_dim, gene_out_dim=self.gene.out_dim, clinical_dim=0, hidden=256)

    def forward(self, img, gene):
        ct_feat = self.ct(img)
        gene_feat = self.gene(gene)
        return self.classifier(ct_feat, gene_feat)

def train_multimodal(img_arrays, gene_arrays, labels, epochs=20, batch_size=16, lr=1e-4, model_name="multimodal"):
    ds = MultiModalDataset(img_arrays=img_arrays, gene_arrays=gene_arrays, labels=labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    model = MultiModalNet(img_channels=1, gene_dim=gene_arrays.shape[1])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    scheduler = ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    writer = SummaryWriter(log_dir=str(MODELS_DIR / f"tb_{model_name}"))
    best_loss = 1e9
    patience = 6
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        for imgs, genes, y in loader:
            imgs = imgs.to(device).float()
            genes = genes.to(device).float()
            y = y.to(device).float()

            # --- FIX: remove extra dimension if it exists ---
            if imgs.dim() == 5 and imgs.size(2) == 1:
                imgs = imgs.squeeze(2)  # shape: [B, C, H, W]

            opt.zero_grad()
            out = model(imgs, genes).squeeze()
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(ds)
        scheduler.step(avg_loss)
        writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f} time={(time.time()-start):.1f}s")

        # early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_best.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    writer.close()
    return model
