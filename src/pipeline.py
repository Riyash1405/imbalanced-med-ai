# src/pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from .models_tabular import get_models

def _onehot_encoder(**kwargs):
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # older sklearn versions use 'sparse' instead of 'sparse_output'
        alt = {}
        for k, v in kwargs.items():
            if k == "sparse_output":
                alt["sparse"] = v
            else:
                alt[k] = v
        return OneHotEncoder(**alt)

RANDOM_STATE = 42

def get_preprocessor(X_df):
    numeric = X_df.select_dtypes(include=['number']).columns.tolist()
    categorical = [c for c in X_df.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(('num', StandardScaler(), numeric))
    if categorical:
        transformers.append(('cat', _onehot_encoder(handle_unknown='ignore', sparse_output=False), categorical))
    return ColumnTransformer(transformers, remainder='drop')

def build_pipeline(X_df, model='rf', resampler='smote'):
    models = get_models()
    clf = models.get(model, models['rf'])

    if resampler == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif resampler == 'smoteenn':
        sampler = SMOTEENN(random_state=RANDOM_STATE)
    else:
        sampler = SMOTE(random_state=RANDOM_STATE)

    pre = get_preprocessor(X_df)
    pipe = ImbPipeline([('pre', pre), ('res', sampler), ('clf', clf)])
    return pipe
