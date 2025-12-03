import numpy as np, torch, time, os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from .multimodal import MultimodalNet
from .loaders import build_multimodal_dataset
from .config import MODELS_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MMDataset(Dataset):
    def __init__(self, imgs, genes, labels):
        self.imgs = imgs.astype('float32'); self.genes = genes.astype('float32'); self.labels = labels.astype('float32')
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.imgs[idx]; 
        if img.ndim==3: img = img[np.newaxis,...]
        return torch.tensor(img), torch.tensor(self.genes[idx]), torch.tensor(self.labels[idx])

def train_multimodal(limit=None, epochs=20, batch_size=8, lr=2e-4):
    ids, imgs, genes, y = build_multimodal_dataset(limit=limit)
    if imgs is None or genes is None:
        print('No paired CT+genomics found. Please add real data under data/ct and data/genomics.')
        N=128; H=224; W=224; G=256
        imgs = np.random.randn(N,1,H,W).astype('float32')
        genes = np.random.randn(N,G).astype('float32')
        y = np.random.randint(0,2,size=N).astype('float32')
    dataset = MMDataset(imgs, genes, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MultimodalNet(gene_dim=genes.shape[1]).to(device)
    opt = Adam(model.parameters(), lr=lr); loss_fn = torch.nn.BCELoss()
    best=1e9; wait=0; pat=6
    for epoch in range(epochs):
        model.train(); total=0; t0=time.time()
        for img, gene, lbl in loader:
            img = img.to(device).float(); gene = gene.to(device).float(); lbl = lbl.to(device).float()
            out = model(img, gene)
            loss = loss_fn(out, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*img.size(0)
        avg = total/len(dataset)
        print(f'Epoch {epoch+1}/{epochs} loss={avg:.4f} time={(time.time()-t0):.1f}s')
        if avg < best:
            best=avg; wait=0; torch.save(model.state_dict(), MODELS_DIR / 'multimodal_best.pt')
        else:
            wait+=1
            if wait>=pat: print('Early stopping'); break
    return MODELS_DIR / 'multimodal_best.pt'
