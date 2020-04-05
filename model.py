from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import pickle
import os, gc, time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Feature creation libraries
from sklearn.random_projection import SparseRandomProjection as sr  # Projection features
from sklearn.cluster import KMeans  # Cluster features
from sklearn.preprocessing import PolynomialFeatures  # Interaction features

root = os.path.dirname(__file__)
path_df = os.path.join(root, 'dataset/heart.csv')
data = pd.read_csv(path_df)

scaler = MinMaxScaler()
train, test = train_test_split(data, test_size=0.25)

X_train = train.drop('target', axis=1)
x_train = train.drop('target', axis=1)
Y_train = train['target']
y_train = train['target']
X_test = test.drop('target', axis=1)
x_test = test.drop('target', axis=1)
Y_test = test['target']
y_test = test['target']

"""Feature engineering """
tmp_train = X_train
tmp_test = X_test
##using statistical methods
feat = ["var", "median", "mean", "std", "max", "min"]
for i in feat:
    X_train[i] = tmp_train.aggregate(i, axis=1)
    X_test[i] = tmp_test.aggregate(i, axis=1)
# Delete not needed variables and release memory
del (tmp_train)
del (tmp_test)
gc.collect()
# So what do we have finally
X_train.shape
X_train.head(1)
X_test.shape
X_test.head(2)
target = Y_train
colNames = X_train.columns.values

##using random projections 
tmp = pd.concat([X_train, X_test],
                axis=0,
                ignore_index=True
                )
NUM_OF_COM = 6

rp_instance = sr(n_components=NUM_OF_COM)
print(rp_instance)
rp = rp_instance.fit_transform(tmp.iloc[:, :13])
rp_col_names = ["r" + str(i) for i in range(6)]

##using Polynomials
degree = 2
poly = PolynomialFeatures(degree,
                          interaction_only=True,
                          include_bias=False)
df = poly.fit_transform(tmp.iloc[:, : 8])
poly_names = ["poly" + str(i) for i in range(36)]

# concatenate all features
tmp = np.hstack([tmp, rp, df])
tmp.shape
X = tmp
del tmp
gc.collect()
y = pd.concat([Y_train, Y_test],
              axis=0,
              ignore_index=True
              )

# Data scaling
# We don't scale targets: Y_test, Y_train as SVC returns the class labels not probability values
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

""" Building the model"""

from sklearn.model_selection import RandomizedSearchCV

# HP tuning using grid search
rf_param_grid = {
    'max_depth': [4, 6, 8, 10],
    'n_estimators': range(1, 30),
    'max_features': ['sqrt', 'auto', 'log2'],
    'min_samples_split': [2, 3, 10, 20],
    'min_samples_leaf': [1, 3, 10, 18],
    'bootstrap': [True, False],

}

rfc = RandomForestClassifier()
models = RandomizedSearchCV(param_distributions=rf_param_grid,
                            estimator=rfc, scoring="accuracy",
                            verbose=0, n_iter=100, cv=5)
# Fitting the models
models.fit(x_train, y_train)

par = models.best_params_
best_model = RandomForestClassifier(n_estimators=par["n_estimators"],
                                    min_samples_split=par['min_samples_split'],
                                    min_samples_leaf=par['min_samples_leaf'],
                                    max_features=par['max_features'],
                                    max_depth=par['max_depth'],
                                    bootstrap=par['bootstrap'])

# Training the best classifier

best_model.fit(x_train, y_train)

# making predictions

y_pred = best_model.predict(x_test)

# Evaluating model accuracy 
from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Saving the trained model for inference
model_path = os.path.join(root, 'models/rfc.sav')
joblib.dump(best_model, model_path)

# Saving the scaler object
scaler_path = os.path.join(root, 'models/scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
