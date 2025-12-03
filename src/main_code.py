import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# ------------------- Imports -------------------
import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# optional: XGBoost & SHAP
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

import joblib

# ------------------- Config -------------------
RANDOM_STATE = 42
REPORT_ROOT = Path("reports")
REPORT_ROOT.mkdir(exist_ok=True)

# ------------------- Helpers -------------------
def save_json(p: Path, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def save_fig(fig, p: Path):
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

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

# ------------------- Data loaders (use your folder structure) -------------------
def load_cardio():
    candidates = [
        Path("data/tabular/cardio_train.csv"),
        Path("data/cardio_train.csv"),
        Path("data/tabular/cardio.csv"),
        Path("data/cardio.csv")
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p, sep=';')
    raise FileNotFoundError("Cardio CSV not found. Add cardio CSV at one of: " + ", ".join(map(str, candidates)))

def load_parkinsons():
    p = Path("data/tabular/parkinsons.csv")
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def load_genomics():
    p = Path("data/genomics/genomics_matrix.csv")
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

# ------------------- Preprocessor & Pipeline builder -------------------
def get_preprocessor(X_df):
    numeric = X_df.select_dtypes(include=['float64','int64','float','int']).columns.tolist()
    categorical = [c for c in X_df.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(('num', StandardScaler(), numeric))
    if categorical:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical))
    return ColumnTransformer(transformers, remainder='drop')

def build_pipeline(X_df, model='rf', resampler='smote'):
    r = resampler.lower()
    if r == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif r == 'smoteenn':
        sampler = SMOTEENN(random_state=RANDOM_STATE)
    else:
        sampler = SMOTE(random_state=RANDOM_STATE)

    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    elif model == 'ada':
        clf = AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE)
    elif model == 'xgb':
        if XGB_AVAILABLE:
            clf = XGBClassifier(n_estimators=200, eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE)
        else:
            clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)

    pre = get_preprocessor(X_df)
    pipe = ImbPipeline([('pre', pre), ('res', sampler), ('clf', clf)])
    return pipe

# ------------------- Core run function -------------------
def run_tabular(df: pd.DataFrame, dataset_name: str, model='rf', resampler='smote', save_plots=True):
    outdir = REPORT_ROOT / dataset_name / f"{model}_{resampler}"
    outdir.mkdir(parents=True, exist_ok=True)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

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
        fig = plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        if save_plots:
            save_fig(fig, outdir / "corr_heatmap.png")

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # pipeline
    pipe = build_pipeline(X_train, model=model, resampler=resampler)

    # fit
    pipe.fit(X_train, y_train)

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
    with open(outdir / "classification_report.txt", "w") as f:
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

    # feature importance
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

    # SHAP for tree models only
    if SHAP_AVAILABLE and hasattr(clf_obj, 'feature_importances_'):
        try:
            print("Computing SHAP values (subset)...")
            X_for_shap = X_test.iloc[:200] if hasattr(X_test, 'iloc') else X_test[:200]
            X_trans = pre.transform(X_for_shap) if pre is not None else np.array(X_for_shap)
            expl = shap.TreeExplainer(clf_obj)
            shap_vals = expl.shap_values(X_trans)
            fig = plt.figure(figsize=(8,6))
            if isinstance(shap_vals, list):
                shap.summary_plot(shap_vals[1], X_trans, show=False)
            else:
                shap.summary_plot(shap_vals, X_trans, show=False)
            if save_plots:
                save_fig(fig, outdir / "shap_summary.png")
        except Exception as e:
            print("SHAP failed:", e)

    print(f"Saved run outputs to: {outdir}")

# ------------------- MAIN -------------------
def main():
    print("Starting main_final.py  (no ADASYN).")
    datasets = []

    # Cardio (try several locations)
    try:
        df_cardio = load_cardio()
        datasets.append(("cardiovascular", df_cardio))
    except Exception as e:
        print("Cardio not found or load error:", e)

    # Parkinsons (from your structure)
    try:
        df_par = load_parkinsons()
        datasets.append(("parkinsons", df_par))
    except Exception as e:
        print("Parkinsons not found or load error:", e)

    # Genomics
    try:
        df_gen = load_genomics()
        datasets.append(("genomics", df_gen))
    except Exception as e:
        print("Genomics not found or load error:", e)

    if len(datasets) == 0:
        print("No datasets found. Place CSVs in data/ as described and re-run.")
        return

    models = ["rf", "xgb", "ada", "lr"]
    resamplers = ["smote", "smoteenn"]

    for ds_name, df in datasets:
        print("\n" + "="*50)
        print("PROCESSING:", ds_name, "| shape:", df.shape)
        print("="*50)
        for m in models:
            for r in resamplers:
                try:
                    run_tabular(df.copy(), ds_name, model=m, resampler=r, save_plots=True)
                except Exception as e:
                    print(f"Run error for {ds_name} {m} {r}: {e}")

    print("\nAll done. Reports saved under:", REPORT_ROOT)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Elapsed time: {:.1f}s".format(time.time() - t0))
