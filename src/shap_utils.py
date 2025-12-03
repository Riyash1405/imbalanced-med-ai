# src/shap_utils.py
import shap
import numpy as np

def compute_shap_for_pipeline(pipe, X_bg_trans, X_shap_trans):
    """
    - pipe: fitted imblearn pipeline with named_steps 'clf' and 'pre'
    - X_bg_trans: numpy array (background) after pre.transform
    - X_shap_trans: numpy array (samples to explain) after pre.transform
    Returns shap_values or raises.
    """
    clf = pipe.named_steps.get('clf', None)
    if clf is None:
        raise ValueError("pipeline has no clf")

    cls_name = clf.__class__.__name__.lower()

    # Tree models: use interventional perturbation to avoid leaf-coverage issues.
    if 'randomforest' in cls_name or 'xgb' in cls_name or 'lightgbm' in cls_name or 'lgbm' in cls_name:
        expl = shap.TreeExplainer(clf, feature_perturbation="interventional")
        # use check_additivity=False to avoid numeric additivity errors on approximations
        vals = expl.shap_values(X_shap_trans, check_additivity=False)
        return vals

    # Linear models
    if 'logistic' in cls_name or 'linear' in cls_name:
        expl = shap.LinearExplainer(clf, X_bg_trans, feature_perturbation="interventional")
        vals = expl.shap_values(X_shap_trans)
        return vals

    # Fallback: KernelExplainer (slow) with small background
    expl = shap.KernelExplainer(clf.predict_proba, X_bg_trans[:50])
    vals = expl.shap_values(X_shap_trans[:50])
    return vals
