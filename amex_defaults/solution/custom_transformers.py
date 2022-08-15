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

NPOLY = 9

class LinRegTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ids, dates):
        # Assuming the dates column is sorted from earlier to later for each customer_ID, and there are no missing values
        self.ids = ids
        self.dates = dates
    def fit(self, X, y=None):
        # X is the column modeled by linear regression
        print(f'fitting {X.name}')
        working = pd.concat([self.dates, X], axis=1).set_index(self.ids).set_axis(['time', 'encoded_col'], axis=1)
        self.fitted_models = []
        def fit_models(group):
            scaler = MinMaxScaler(feature_range=(0, 1))
            time = scaler.fit_transform((group['time'] - group['time'].iloc[0]).dt.days.values.reshape(-1, 1)).reshape(1, -1)[0]
            regX = pd.DataFrame()
            for i in range(NPOLY):
                regX[f'poly{i}'] = time ** i
            model = LinearRegression(fit_intercept=False, n_jobs=-1)
            model.fit(regX, group['encoded_col'])
            self.fitted_models.append(model)
            if len(self.fitted_models) % 100000 == 0:
                print(f'{len(self.fitted_models)}')
        working.groupby(working.index).apply(fit_models)
        print(f'fitted {X.name}')
        return self
    def transform(self, X, y=None):
        print(f'transforming {X.name}')
        uniques = self.ids.unique()

        from io import StringIO
        from csv import writer
        
        output = StringIO()
        csv_writer = writer(output)

        sample_coefs = self.fitted_models[0].coef_
        # write column header
        csv_writer.writerow([f'{X.name}_coef{i}' for i in range(len(sample_coefs))])
        for i in range(len(uniques)):
            csv_writer.writerow(self.fitted_models[i].coef_)
        
        output.seek(0)
        ret = pd.read_csv(output)
        return ret.set_index(uniques)