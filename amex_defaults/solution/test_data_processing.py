import pandas as pd
from xgboost import XGBClassifier

# uncached
# my_model = XGBClassifier(n_jobs=-1)
# my_model.fit(X_train_input, y_train['target'])
# my_model.save_model('../data/solution/fitted_model.json')

# cached
my_model = XGBClassifier()
my_model.load_model('../data/solution/fitted_model.json')
print("model loaded")

cols = ['customer_ID', 'S_2', 'D_77', 'P_2', 'D_48', 'D_61', 'B_17', 'D_62', 'D_44', 'B_9', 'D_75', 'B_18']
X_test = pd.read_csv('../data/test_data.csv', usecols=cols)
X_test['S_2'] = pd.to_datetime(X_test['S_2'], format='%Y-%m-%d')
print('raw dataset loaded')

# uncached
from preprocessing import preprocess
X_test_preprocessed = preprocess(X_test)
X_test_preprocessed.to_csv('../data/solution/imputed_test_data.csv')
print('data preprocessed')

# uncached
from preprocessing import encode
X_test_input = encode(X_test_preprocessed)
X_test_input.to_csv('../data/solution/encoded_test_data.csv')
print('data encoded')

result = pd.concat([X_test.customer_ID.unique(), my_model.predict(X_test_input)], axis=1)
result.to_csv('../data/solution/predictions.csv')
print("data predicted")