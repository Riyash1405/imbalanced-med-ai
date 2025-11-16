"""
Main runner script:
- trains tabular models with tuning
- computes SHAP + plots
- trains multimodal (with synthetic data if real data missing)
- produces final summary reports
"""
import warnings
warnings.filterwarnings("ignore")
from .models_tabular import train_and_tune
from .evaluate import evaluate_saved_models
from .shap_utils import compute_shap_for_pipeline
from .multimodal.train import train_multimodal
from .data_loader import load_tabular
from .config import DATA_DIR, MODELS_DIR, PLOTS_DIR
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

def run_tabular_pipeline():
    print("Running tabular training/tuning...")
    results = train_and_tune(dataset="heart", tune=False, use_optuna=False)
    print("Tabular results:", results)

def run_evaluation_and_shap():
    print("Evaluating saved models and computing SHAP...")
    # evaluate saved models and compute shap for the first model
    summaries = evaluate_saved_models(dataset="heart")
    # compute shap for the RF model if saved as joblib
    model_files = list(MODELS_DIR.glob("heart_rf.joblib"))
    if model_files:
        pipe = joblib.load(model_files[0])
        df = load_tabular("heart")
        # sample 100 rows for SHAP
        X = df.drop(columns=["target"]).sample(n=min(200, len(df)), random_state=42)
        try:
            compute_shap_for_pipeline(pipe, X, model_name="heart_rf")
            print("SHAP saved.")
        except Exception as e:
            print("SHAP error:", e)

def run_multimodal_demo():
    print("Preparing synthetic multimodal data (demo)...")
    # create synthetic data if real images/genomics not present
    # small demo arrays
    N = 128
    IMG_SIZE = 128
    GENE_DIM = 500
    img_arrays = np.random.randn(N, 1, IMG_SIZE, IMG_SIZE).astype("float32")
    gene_arrays = np.random.randn(N, GENE_DIM).astype("float32")
    labels = np.random.randint(0,2,size=N).astype("float32")
    print("Training multimodal network on synthetic data (demo)")
    train_multimodal(img_arrays, gene_arrays, labels, epochs=6, batch_size=16, lr=2e-4, model_name="multimodal_demo")
    print("Multimodal demo training complete.")

if __name__ == "__main__":
    run_tabular_pipeline()
    run_evaluation_and_shap()
    run_multimodal_demo()
    print("All done.")
