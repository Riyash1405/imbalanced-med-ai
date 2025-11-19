import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from .config import MODELS_DIR, REPORTS_DIR, PLOTS_DIR
from .data_loader import load_tabular, train_test_split_tabular
from .utils import plot_roc, plot_pr, save_json
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

def evaluate_saved_models(dataset="heart"):
    models = list(MODELS_DIR.glob(f"{dataset}_*.joblib"))
    df = load_tabular(dataset)
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    summaries = []
    for m in models:
        name = m.stem
        print("Evaluating", name)
        pipe = joblib.load(m)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        summaries.append({"model":name, "auc":auc, "f1":f1, "confusion":cm, "report":report})
        plot_roc(name, y_test, y_prob, PLOTS_DIR / "roc_curves" / f"{name}_roc.png")
        plot_pr(name, y_test, y_prob, PLOTS_DIR / "pr_curves" / f"{name}_pr.png")
    save_json(summaries, REPORTS_DIR / f"{dataset}_summary.json")
    return summaries
