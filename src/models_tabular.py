# src/models_tabular.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42

def get_models():
    models = {
        "rf": RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        "lr": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    }
    if XGB_AVAILABLE:
        models["xgb"] = XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=RANDOM_STATE)
    else:
        # fallback
        models["xgb"] = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    return models
