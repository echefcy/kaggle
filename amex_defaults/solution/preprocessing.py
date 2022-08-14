import pandas as pd
from custom_transformers import *

# top 10 mi scores: ['D_77', 'P_2', 'D_48', 'D_61', 'B_17', 'D_62', 'D_44', 'B_9', 'D_75', 'B_18']

SELECTED_FEATURES = ['D_77', 'P_2', 'D_48', 'D_61', 'B_17', 'D_62', 'D_44', 'B_9', 'D_75', 'B_18']

def preprocess_floats(data: pd.DataFrame):
    # preprocess(data) is the preprocessed float columns
    # data is the complete feature set with customer_ID as its index
    float_cols = data.select_dtypes('float64')
    imputer = FloatImputer()
    float_cols = imputer.fit_transform(float_cols)
    return float_cols

def preprocess_cats(data: pd.DataFrame):
    # preprocess(data) is the preprocessed categorical columns
    # data is the complete feature set with customer_ID as its index
    discrete_cols = data.select_dtypes(exclude='float64')
    for cname in discrete_cols.columns:
        discrete_cols[cname], _ = discrete_cols[cname].factorize()
    return discrete_cols

def preprocess(data: pd.DataFrame):
    data = data.set_index('customer_ID')
    selected_features = SELECTED_FEATURES
    selected_data = data[selected_features]

    cats = preprocess_cats(selected_data)
    floats = preprocess_floats(selected_data)

    columns = [data[['S_2']], cats, floats]

    return pd.concat(columns, axis=1)

def encode(data: pd.DataFrame):
    # assumes data was preprocessed
    linreg = LinRegTransformer(data.index, data['S_2'])
    floats = data.select_dtypes('float64')
    cats = data.select_dtypes(['object', 'int64', 'category'])

    columns = []
    for cname in floats.columns:
        regressed = linreg.fit_transform(floats[cname])
        columns.append(regressed)

    # mean encode the categoricals
    folder = IDMeanTransformer()
    for cname in cats.columns:
        folded = folder.fit_transform(cats[cname])
        columns.append(folded)
    
    return pd.concat(columns, axis=1)

def preprocess_encode(data: pd.DataFrame):
    data = data.set_index('customer_ID')
    selected_features = SELECTED_FEATURES
    selected_data = data[selected_features]

    cats = preprocess_cats(selected_data)
    floats = preprocess_floats(selected_data)

    # linear regression encode the floats
    linreg = LinRegTransformer(data.index, data['S_2'])
    columns = []
    for cname in floats.columns:
        regressed = linreg.fit_transform(floats[cname])
        columns.append(regressed)

    # mean encode the categoricals
    folder = IDMeanTransformer()
    for cname in cats.columns:
        folded = folder.fit_transform(cats[cname])
        columns.append(folded)
    
    return pd.concat(columns, axis=1)