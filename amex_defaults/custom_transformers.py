# grouping on index
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ByIndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, cnames = None):
        self.cnames = cnames
        self.transformer = transformer
    def fit(self, X: pd.DataFrame, y = None):
        self.fitted_transformers = X.groupby(X.index).apply(
            lambda df: self.transformer.fit(df if self.cnames == None else df[self.cnames]))
        self.fitted_transformers = {self.fitted_transformers.index[i] : self.fitted_transformers.iloc[i] for i in range(len(self.fitted_transformers))}
        return self
    def transform(self, X: pd.DataFrame, y = None):
        transformed = X.groupby(X.index).apply(
            lambda df: pd.DataFrame(
                self.fitted_transformers[df.index[0]]
                .transform(df if self.cnames == None else df[self.cnames])
            )
        )
        transformed.reset_index()
        transformed.index = X.index
        return transformed

class IDMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_name = None):
        self.id_name = id_name
    def fit(self, X: pd.DataFrame, y = None):
        return self
    def transform(self, X: pd.DataFrame, y = None):
        return X.groupby(X.index if self.id_name == None else self.id_name).apply(lambda df: df.mean()).unstack().reset_index()
        
###
# class DFTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, cnames)

# # class EncoderWrapper(BaseEstimator, TransformerMixin):
# #     def __init__(self, encoder):
# #         self.encoder = encoder
# #     def fit(self, X: np.ndarray, y = None):
# #         self.encoder.fit(X, y)
# #         return self
# #     def transform(self, X: np.ndarray, y = None):
# #         encoded_cols = self.encoder.transform(X)
# #         encoded_df = pd.DataFrame(encoded_cols)
# #         return encoded_df

# def column_types(features: pd.DataFrame):
#     categoricals = []
#     objects = []
#     numericals = []
#     for cname in features.columns:
#         col = features[cname]
#         if col.dtype == 'object':
#             if col.nunique() < 10:
#                 categoricals.append(cname)
#             else:
#                 objects.append(cname)
#         elif col.dtype in ['int64', 'float64']:
#             numericals.append(cname)
#         else:
#             print(f"{cname}: no column category matched")
#     return categoricals, numericals, objects

# def preprocess(features: pd.DataFrame, y = None):
#     categoricals, numericals, objects = column_types(features)
    
#     # Preprocesses categoricals
#     onehot = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
#     ])
#     transformers = ColumnTransformer(transformers=[
#         ('cat', onehot, categoricals)
#     ], remainder='passthrough')
    
#     encoded_cols = transformers.fit_transform(features)
#     df = pd.DataFrame(encoded_cols)
#     print(encoded_cols)

#     df = df.groupby('customer_ID').apply(lambda df: df.mean()).unstack().reset_index()

#     return df

# # X = sample2.drop('D_64', axis=1)
# # X = preprocess(X)
# # X.head()
###