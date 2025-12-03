# src/run_tabular.py
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Force non-GUI matplotlib backend early
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
import joblib

from .pipeline import build_pipeline
from .utils import save_json, save_fig
from .shap_utils import compute_shap_for_pipeline

# try import shap to check availability
try:
    import shap  # noqa
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

REPORT_ROOT = Path("reports")
REPORT_ROOT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def expected_calibration_error(y_true, y_proba, n_bins=10):
    if y_proba is None:
        return None
    bins = np.linspace(0,1,n_bins+1)
    binids = np.digitize(y_proba, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_proba[mask].mean()
        ece += (mask.sum()/len(y_true)) * abs(acc - conf)
    return float(ece)

def run_tabular(df: pd.DataFrame, dataset_name: str, model='rf', resampler='smote', save_plots=True):
    outdir = REPORT_ROOT / dataset_name / f"{model}_{resampler}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Drop id if present
    if 'id' in df.columns:
        try:
            df = df.drop(columns=['id'])
        except Exception:
            pass

    # Ensure target named 'target'
    if 'target' not in df.columns:
        df = df.rename(columns={df.columns[-1]: 'target'})

    # Coerce numeric-like strings to numeric
    for c in df.columns:
        if df[c].dtype == object:
            try:
                # remove thousands separators and surrounding whitespace
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='ignore')
            except Exception:
                pass

    target_col = 'target'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ensure integer labels
    try:
        y = y.astype(int)
    except Exception:
        # if labels are strings like 'yes'/'no', try mapping
        uniques = list(pd.Series(y).unique())
        mapping = {v: i for i, v in enumerate(sorted(uniques))}
        y = pd.Series(y).map(mapping).astype(int)

    print(f"\n{dataset_name} | {model} + {resampler}  — features {X.shape}  samples {len(df)}")
    save_json(outdir / "dataset_info.json", {
        "dataset": dataset_name, "n_samples": int(len(df)), "n_features": int(X.shape[1]), "target_col": target_col
    })

    # target distribution
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title("Target distribution (original)")
    if save_plots:
        save_fig(fig, outdir / "target_dist_before.png")

    # correlation heatmap
    if X.shape[1] <= 60:
        try:
            fig = plt.figure(figsize=(10,8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            if save_plots:
                save_fig(fig, outdir / "corr_heatmap.png")
        except Exception:
            pass

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # pipeline
    pipe = build_pipeline(X_train, model=model, resampler=resampler)

    # fit
    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        print("Pipeline fit failed:", e)
        return

    # predict
    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_proba = pipe.predict_proba(X_test)[:,1]
    except Exception:
        y_proba = None

    # metrics
    acc = float(accuracy_score(y_test, y_pred))
    rpt = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    pr_prec, pr_recall, _ = precision_recall_curve(y_test, y_proba if y_proba is not None else np.zeros_like(y_pred))
    pr_auc = float(auc(pr_recall, pr_prec)) if y_proba is not None else None
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    ece = expected_calibration_error(np.array(y_test), np.array(y_proba) if y_proba is not None else None)

    metrics = {"accuracy": acc, "roc_auc": roc_auc, "pr_auc": pr_auc, "ece": ece, "confusion_matrix": cm, "classification_report": rpt}
    save_json(outdir / "metrics.json", metrics)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred))

    joblib.dump(pipe, outdir / "model.pkl")

    # confusion matrix plot
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    if save_plots:
        save_fig(fig, outdir / "confusion_matrix.png")

    # ROC / PR / calibration
    if y_proba is not None:
        from sklearn.metrics import RocCurveDisplay
        fig = plt.figure(figsize=(6,5))
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve")
        if save_plots:
            save_fig(fig, outdir / "roc_curve.png")

        fig = plt.figure(figsize=(6,5))
        plt.plot(pr_recall, pr_prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve (AUC={pr_auc:.4f})" if pr_auc else "PR Curve")
        if save_plots:
            save_fig(fig, outdir / "pr_curve.png")

        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        fig = plt.figure(figsize=(6,5))
        plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--', color='gray')
        plt.title(f"Calibration curve (ECE={ece:.4f})" if ece is not None else "Calibration")
        if save_plots:
            save_fig(fig, outdir / "calibration_curve.png")

    # feature names extraction
    clf_obj = pipe.named_steps.get('clf', None)
    pre = pipe.named_steps.get('pre', None)
    feature_names = []
    try:
        if pre is not None and hasattr(pre, 'transformers_'):
            num_feats=[]; cat_feats=[]
            for name, trans, cols in pre.transformers_:
                if name=='num': num_feats = cols
                if name=='cat': cat_feats = cols
            cat_out=[]
            try:
                enc = pre.named_transformers_['cat']
                if hasattr(enc, 'get_feature_names_out'):
                    cat_out = enc.get_feature_names_out(cat_feats).tolist()
                else:
                    cat_out = cat_feats
            except Exception:
                cat_out = cat_feats
            feature_names = list(num_feats) + list(cat_out)
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
    except Exception:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # feature importance
    if hasattr(clf_obj, 'feature_importances_'):
        try:
            importances = clf_obj.feature_importances_
            idx = np.argsort(importances)[::-1][:30]
            fig = plt.figure(figsize=(8,6))
            sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
            plt.title("Feature Importances (top 30)")
            if save_plots:
                save_fig(fig, outdir / "feature_importance.png")
        except Exception:
            pass

    # SHAP
    if SHAP_AVAILABLE:
        try:
            print("Computing SHAP values (subset)...")
            # Build background & sample for SHAP
            if pre is not None:
                # use transformed background from training data (larger than before)
                bg_n = min(500, len(X_train))
                X_bg = X_train.sample(bg_n, random_state=RANDOM_STATE)
                X_bg_trans = pre.transform(X_bg)

                shap_n = min(200, len(X_test))
                X_shap = X_test.sample(shap_n, random_state=RANDOM_STATE)
                X_shap_trans = pre.transform(X_shap)
            else:
                X_bg_trans = X_train.iloc[:min(500, len(X_train))].values
                X_shap_trans = X_test.iloc[:min(200, len(X_test))].values

            shap_vals = compute_shap_for_pipeline(pipe, X_bg_trans, X_shap_trans)

            if shap_vals is not None:
                # summary plot
                try:
                    if isinstance(shap_vals, list):
                        shap.summary_plot(shap_vals[1], X_shap_trans, feature_names=feature_names, show=False)
                    else:
                        shap.summary_plot(shap_vals, X_shap_trans, feature_names=feature_names, show=False)
                    fig = plt.gcf()
                    if save_plots:
                        save_fig(fig, outdir / "shap_summary.png")
                except Exception as e:
                    print("SHAP plotting error:", e)
        except Exception as e:
            print("SHAP failed:", e)

    print(f"Saved run outputs to: {outdir}")

def run_all_tabular(datasets):
    models = ["rf", "xgb", "lr"]
    resamplers = ["smote", "smoteenn"]
    for ds_name, df in datasets:
        print("\n" + "="*50)
        print("PROCESSING:", ds_name, "| shape:", df.shape)
        print("="*50)
        for m in models:
            for r in resamplers:
                try:
                    run_tabular(df.copy(), dataset_name=ds_name, model=m, resampler=r, save_plots=True)
                except Exception as e:
                    print(f"Run error for {ds_name} {m} {r}: {e}")
