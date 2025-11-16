"""
Tabular preprocessing and pipeline builders.
Includes many resampling options and model builders.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# resamplers
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

def get_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers=transformers)

def build_model_pipeline(model_name="rf", resampler="smote"):
    models = {
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "ada": AdaBoostClassifier(n_estimators=200),
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm": SVC(probability=True)
    }
    resamplers = {
        "smote": SMOTE(),
        "adasyn": ADASYN(),
        "borderline": BorderlineSMOTE(),
        "smotetomek": SMOTETomek(),
        "smoteenn": SMOTEENN()
    }
    model = models[model_name]
    sampler = resamplers.get(resampler, SMOTE())
    # preprocessor will be attached later (needs X to compute cols)
    pipe = ImbPipeline(steps=[
        ("preprocess", "passthrough"),  # will replace with ColumnTransformer in training code
        ("resample", sampler),
        ("clf", model)
    ])
    return pipe
