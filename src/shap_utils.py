import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import PLOTS_DIR
from sklearn.base import is_classifier

def compute_shap_for_pipeline(pipe, X_sample, model_name="model", max_display=20):
    """
    pipe: full pipeline with preprocess + clf
    X_sample: pandas DataFrame of raw features (not transformed)
    """
    preproc = pipe.named_steps["preprocess"]
    model = pipe.named_steps["clf"]
    X_trans = preproc.transform(X_sample)

    # For tree models use TreeExplainer on model only but shap expects original feature names
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        # use X_sample for feature names if preproc has transformer that returns numpy; we assume columns match in order
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        plt.savefig(PLOTS_DIR / "shap" / f"{model_name}_summary.png")
        plt.close()
    except Exception as e:
        print("TreeExplainer failed, falling back to KernelExplainer:", e)
        # Kernel explainer (slower)
        background = X_trans[np.random.choice(X_trans.shape[0], min(50, X_trans.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1], background)
        shap_values = explainer.shap_values(X_trans[:100])
        shap.summary_plot(shap_values, X_sample.iloc[:100], show=False)
        plt.savefig(PLOTS_DIR / "shap" / f"{model_name}_kernel_summary.png")
        plt.close()
