"""
Multimodal models: CNN for CT, MLP for clinical/genomic, Transformer encoder for gene sequences.
Fusion and classifier head included.
PyTorch-based implementation.
"""
import torch
import torch.nn as nn
import math

class CTEncoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=16, img_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.out_dim = base_filters*4

    def forward(self, x):
        return self.encoder(x)

class GeneTransformer(nn.Module):
    def __init__(self, input_dim=500, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.embed(x).unsqueeze(0)  # seq_len=1, batch, d_model
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, d_model)
        return x

class FusionClassifier(nn.Module):
    def __init__(self, ct_out_dim, gene_out_dim, clinical_dim=0, hidden=256):
        super().__init__()
        fusion_dim = ct_out_dim + gene_out_dim + clinical_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden//2),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )

    def forward(self, ct_feat, gene_feat, clinical=None):
        if clinical is not None:
            x = torch.cat([ct_feat, gene_feat, clinical], dim=1)
        else:
            x = torch.cat([ct_feat, gene_feat], dim=1)
        return self.fc(x)
