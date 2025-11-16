# MedImbalance-Ensemble (Full Project) 🚀

Ensemble learning + multimodal fusion pipeline for imbalanced medical diagnosis.
This repository contains:
- Robust tabular ML pipelines (SMOTE/ADASYN/SMOTEENN, RandomForest, AdaBoost, LR, SVM)
- Hyperparameter tuning (GridSearch, Optuna optional)
- SHAP explainability + publication-quality plots (ROC, PR, Calibration)
- Multimodal deep-learning (CNN for CT, Transformer for genomics) with early-stopping and tensorboard
- Report generation and model saving

## Quick start (Windows, VS Code)
1. Clone repo & open in VS Code.
2. Create & activate venv:
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1

(or activate for cmd)
3. Install deps:

pip install -r requirements.txt


Use --user if permission issues.
4. Place heart.csv and parkinsons.csv files in data/.
5. Run:

python -m src.main


Check outputs in outputs/ and models in models/.

Project layout

(see repository root listing)

Notes

Multimodal component uses synthetic data by default. Replace with CT images / genomics by populating data/ct_images/ and data/genomics/.

For SHAP and KernelExplainer large datasets, computations may be slow — sample a subset to visualize.

Authors

Group 14 — Riyash Patel, Karmvirsinh Parmar, Vinit Mepani

License

Academic use recommended. Cite appropriately.


---

# How to run (step-by-step, VS Code / Windows)
1. Open project folder in VS Code.  
2. Create virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1


If PowerShell policy prevents activation:

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1


Upgrade pip and install packages:

python -m pip install --upgrade pip
pip install -r requirements.txt


If permission errors occur, append --user or use the venv above.

Create outputs subfolders (if not created):

models outputs outputs/plots outputs/plots/roc_curves outputs/plots/pr_curves outputs/plots/shap outputs/reports


The code will create base directories but creating subfolders avoids permission errors.

Run the project:

python -m src.main


After run, inspect:

models/ for saved joblib/pt models

outputs/reports/ for text reports and JSON summaries

outputs/plots/roc_curves/ and pr_curves/ for plots

TensorBoard logs: open models/tb_multimodal_demo with tensorboard --logdir models and browse http://localhost:6006/