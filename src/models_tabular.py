"""
Large training and tuning module for tabular models.
Includes grid search and Optuna tuning.
Saves best models and produces detailed evaluation files.
"""
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss
)
from .pipelines import get_preprocessor, build_model_pipeline
from .data_loader import train_test_split_tabular, load_tabular
from .config import MODELS_DIR, REPORTS_DIR, PLOTS_DIR, RANDOM_STATE
from .utils import plot_roc, plot_pr, save_model, save_json
import matplotlib.pyplot as plt
import seaborn as sns

# optuna for Bayesian tuning
try:
    import optuna
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False

def train_and_tune(dataset="heart", tune=False, use_optuna=False):
    # Ensure main directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "roc_curves").mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "pr_curves").mkdir(parents=True, exist_ok=True)
    (PLOTS_DIR / "confusion").mkdir(parents=True, exist_ok=True)

    df = load_tabular(dataset)
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    preproc = get_preprocessor(X_train)

    # list of models to try
    candidate_models = [
        ("rf", "smote"),
        ("ada", "smote"),
        ("lr", "smoteenn"),
        ("svm", "smote")
    ]

    results = {}

    for (mname, sampler) in candidate_models:
        print(f"\n=== TRAINING {mname} with {sampler} ===")
        pipe = build_model_pipeline(model_name=mname, resampler=sampler)
        # replace placeholder preprocess
        pipe.steps[0] = ("preprocess", preproc)

        # quick grid for rf/ada/lr
        if mname == "rf":
            param_grid = {"clf__n_estimators":[200,300], "clf__max_depth":[None,10,20]}
        elif mname == "ada":
            param_grid = {"clf__n_estimators":[100,200], "clf__learning_rate":[0.5,1.0]}
        elif mname == "lr":
            param_grid = {"clf__C":[0.01,0.1,1.0]}
        else:
            param_grid = {}

        if tune and param_grid:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            g = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            g.fit(X_train, y_train)
            best = g.best_estimator_
            print("Best params:", g.best_params_)
        else:
            pipe.fit(X_train, y_train)
            best = pipe

        # save model
        fname = MODELS_DIR / f"{dataset}_{mname}.joblib"
        save_model(best, fname)

        # predict on test
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:,1]

        report = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)

        results[mname] = {"auc":auc, "f1":f1, "brier":brier, "report":report}

        # save textual report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpath = REPORTS_DIR / f"{dataset}_{mname}_{timestamp}.txt"
        with open(rpath, "w") as f:
            f.write(report_str)
            f.write(f"\nAUC: {auc}\nF1: {f1}\nBrier: {brier}\nConfusion:\n{cm}\n")

        # plots
        plot_roc(f"{dataset}_{mname}", y_test, y_prob, PLOTS_DIR / "roc_curves" / f"{dataset}_{mname}_roc.png")
        plot_pr(f"{dataset}_{mname}", y_test, y_prob, PLOTS_DIR / "pr_curves" / f"{dataset}_{mname}_pr.png")

        # confusion matrix heatmap
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion - {dataset}_{mname}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion" / f"{dataset}_{mname}_confusion.png")
        plt.close()

    return results
