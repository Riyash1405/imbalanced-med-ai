import torch, numpy as np, matplotlib.pyplot as plt, os
from .multimodal import MultimodalNet
from .loaders import build_multimodal_dataset
from .config import MODELS_DIR, PLOTS_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def grad_cam(model, img_tensor, gene_tensor):
    model.eval(); img_tensor = img_tensor.clone().detach().to(device).unsqueeze(0).float(); img_tensor.requires_grad_(True)
    out = model(img_tensor, gene_tensor.unsqueeze(0).to(device))
    out.backward()
    grad = img_tensor.grad.cpu().numpy()[0,0]
    cam = np.maximum(grad, 0); cam = (cam - cam.min()) / (cam.max()-cam.min()+1e-8)
    return cam
def generate_heatmaps(limit=5):
    model = MultimodalNet().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / 'multimodal_best.pt', map_location=device))
    ids, imgs, genes, y = build_multimodal_dataset(limit=limit)
    if ids is None:
        print('No real data to generate heatmaps.')
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for i in range(min(limit, len(ids))):
        img = imgs[i]; gene = genes[i]
        img_t = torch.tensor(img).unsqueeze(0) if img.ndim==3 else torch.tensor(img).unsqueeze(0)
        gene_t = torch.tensor(gene)
        cam = grad_cam(model, img_t.squeeze(0), gene_t)
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.imshow(img[0], cmap='gray'); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(img[0], cmap='gray'); plt.imshow(cam, cmap='jet', alpha=0.5); plt.axis('off')
        plt.savefig(os.path.join(PLOTS_DIR, f'heatmap_{ids[i]}.png')); plt.close()
    print('Heatmaps saved to', PLOTS_DIR)
