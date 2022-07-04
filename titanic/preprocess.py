import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def test_preprocess(data, features):
    X = data[features]

    # Encode
    enc_cols = ['Sex']
    oh_enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    oh_cols = pd.DataFrame(oh_enc.fit_transform(X[enc_cols]))
    oh_cols.columns = oh_enc.get_feature_names_out()
    oh_cols.index = X.index

    X_enc = X.drop(enc_cols, axis=1)
    X_enc = pd.concat([X_enc, oh_cols], axis=1)

    # Impute
    imp = SimpleImputer(missing_values=pd.NA, strategy="mean")
    X_processed = pd.DataFrame(imp.fit_transform(X_enc))
    X_processed.columns = X_enc.columns

    return X_processed


def train_preprocess(data, features):
    # Split
    y = data.Survived
    X = data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # Encode
    enc_cols = ['Sex']
    oh_enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    oh_train_cols = pd.DataFrame(oh_enc.fit_transform(train_X[enc_cols]))
    oh_train_cols.columns = oh_enc.get_feature_names_out()
    oh_valid_cols = pd.DataFrame(oh_enc.transform(val_X[enc_cols]))
    oh_valid_cols.columns = oh_enc.get_feature_names_out()
    oh_train_cols.index = train_X.index
    oh_valid_cols.index = val_X.index

    train_X_enc = train_X.drop(enc_cols, axis=1)
    val_X_enc = val_X.drop(enc_cols, axis=1)
    train_X_enc = pd.concat([train_X_enc, oh_train_cols], axis=1)
    val_X_enc = pd.concat([val_X_enc, oh_valid_cols], axis=1)

    # Impute
    imp = SimpleImputer(missing_values=pd.NA, strategy="mean")
    train_X_processed = pd.DataFrame(imp.fit_transform(train_X_enc))
    val_X_processed = pd.DataFrame(imp.transform(val_X_enc))
    train_X_processed.columns = train_X_enc.columns
    val_X_processed.columns = val_X_enc.columns

    return train_X_processed, val_X_processed, train_y, val_y
