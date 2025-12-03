from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessor(X):
    numeric = X.select_dtypes(
        include=['float64','int64','float','int']
    ).columns.tolist()

    categorical = [c for c in X.columns if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(('num', StandardScaler(), numeric))
    if categorical:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical))

    return ColumnTransformer(transformers, remainder='drop')
