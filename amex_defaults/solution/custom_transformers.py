# grouping on index
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class IDMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_name = None):
        self.id_name = id_name
    def fit(self, X, y = None):
        return self
    def transform(self, X: pd.DataFrame, y = None):
        transformed = X.groupby(X.index if self.id_name == None else self.id_name).apply(lambda df: df.mean())
        return transformed

class FloatImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
        
    def fill_group(self, group, col_defaults):
        for cname in group.columns:
            if group[cname].isna().all():
                group[cname].fillna(col_defaults[cname], inplace=True)

    def fit(self, X: pd.DataFrame, y = None):
        self.fitted_imputers = {}
        col_defaults = {cname : 0.0 for cname in X.columns}
        def fit_group(group: pd.DataFrame):
            self.fill_group(group, col_defaults)
            imp = SimpleImputer()
            self.fitted_imputers[group.index[0]] = imp.fit(group)
        X.groupby(X.index).apply(fit_group)
        return self
        
    def transform(self, X: pd.DataFrame, y = None):
        col_defaults = {cname : X[cname].mean() for cname in X.columns}
        def transform_group(group: pd.DataFrame):
            self.fill_group(group, col_defaults)
            return pd.DataFrame(self.fitted_imputers[group.index[0]].transform(group))
        transformed = X.groupby(X.index).apply(transform_group)
        return transformed.reset_index().set_index('customer_ID').drop('level_1', axis=1).set_axis(X.columns, axis=1)

class LinRegTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ids, dates):
        # Assuming the dates column is sorted from earlier to later for each customer_ID, and there are no missing values
        self.ids = ids
        self.dates = dates
    def fit(self, X, y=None):
        # X is the column modeled by linear regression
        working = pd.DataFrame(self.dates).set_index(self.ids).set_axis(['time'], axis=1)
        self.fitted_models = {}
        def fit_models(group):
            scaler = MinMaxScaler(feature_range=(0, 1))
            time = scaler.fit_transform((group['time'] - group['time'].iloc[0]).dt.days.values.reshape(-1, 1)).reshape(1, -1)[0]
            regX = pd.DataFrame()
            for i in range(11):
                regX[f'poly{i}'] = time ** i
            model = LinearRegression(fit_intercept=False)
            model.fit(regX, group)
            self.fitted_models[group.index[0]] = model
        working.groupby(working.index).apply(fit_models)
        return self
    def transform(self, X, y=None):
        uniques = self.ids.unique()
        
        # This sets up the shape of the dataframe
        sample_coefs = self.fitted_models[uniques[0]].coef_[0]
        ret = pd.DataFrame(columns=[f'{X.name}_coef{i}' for i in range(len(sample_coefs))])
        for i in range(len(uniques)):
            ret.loc[uniques[i]] = self.fitted_models[uniques[i]].coef_[0]
        return ret