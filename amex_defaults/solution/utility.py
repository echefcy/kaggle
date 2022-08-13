import pandas as pd
import numpy as np
from mlxtend.preprocessing import minmax_scaling

def get_sample(features, size):
    # gets (size) amount of customers
    prng = np.random.RandomState(0)
    g = features.groupby('customer_ID')

    portion = features[g.ngroup().isin(prng.choice(g.ngroups, size, replace=False))]
    return portion

def get_y(ids, labels):
    label_lookup = dict()
    for id in ids:
        if id not in label_lookup:
            target = labels.loc[labels.customer_ID == id, 'target']
            label_lookup[id] = target.iloc[0]

    return pd.Series([label_lookup[id] for id in ids], index=ids)

def get_binary_cols(features):
    obj_cols = features.select_dtypes(include=['O', '<M8[ns]'])
    scaled_data = minmax_scaling(features.drop(obj_cols, axis=1), columns=features.drop(obj_cols, axis=1).columns)
    binary_cols = []
    for cname in scaled_data:
        cmean = scaled_data[cname].mean()
        if len(scaled_data.loc[(0.01 < scaled_data[cname]) & (scaled_data[cname] < 0.99)]) == 0:
            binary_cols.append(cname)
    return binary_cols