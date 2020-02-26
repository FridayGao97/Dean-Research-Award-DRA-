import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp

from sklearn.linear_model import Perceptron      # Perceptron toolbox
from sklearn.neural_network import MLPRegressor  # MLP toolbox

from sklearn import datasets 
from sklearn.neural_network import MLPClassifier 
from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('EdmontonRealEstateData.csv')
df = df.set_index('taxroll_number')

sdf = pd.read_csv('EdmontonRealEstateData.csv')
sdf = sdf.set_index('taxroll_number')
df.head()

#sns.distplot(df['assessed_value'])


y = df.assessed_value

df = df.drop('assessed_value', axis=1)
all_df = df.append(sdf)
all_df.shape

all_features = list(df.columns.values)
numeric_features = list(df.select_dtypes(include=[np.number]).columns.values)
categorical_features = [f for f in all_features if not(f in numeric_features)]

numeric_df = all_df[numeric_features]
numeric_df.shape

X = numeric_df.as_matrix()
imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X)
X = imp.transform(X)

scaler = pp.StandardScaler()
#Todo: Fit and transform data using scaler
scaler = scaler.fit_transform(X)
X[0, :]
def process_categorical(ndf, df, categorical_features):
    for f in categorical_features:
        new_cols = pd.DataFrame(pd.get_dummies(df[f]))
        new_cols.index = df.index
        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
    return ndf

numeric_df = pd.DataFrame(X)
numeric_df.index = all_df.index
combined_df = process_categorical(numeric_df, all_df, categorical_features)
X = combined_df.as_matrix()


pft = PolynomialFeatures(degree = 2)
X_poly = pft.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y,test_size = 0.40,random_state = 42)
model = linear_model.Ridge(alpha = 300)
# alpha is the regularization parameter(don't get confused by the symbol)
model.fit(X_train, y_train)
predictionTestSet = model.predict(X_test)
from sklearn.metrics import mean_squared_error
errorTestSet = mean_squared_error(y_test, predictionTestSet)
print("Error in test set: {:.2f}\n".format(errorTestSet))







# scaler = pp.StandardScaler()
# #Todo: Fit and transform data using scaler
# scaler = scaler.fit_transform(X)
# X[0, :]

# def process_categorical(ndf, df, categorical_features):
#     for f in categorical_features:
#         new_cols = pd.DataFrame(pd.get_dummies(df[f]))
#         new_cols.index = df.index
#         ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
#     return ndf

# numeric_df = pd.DataFrame(X)
# numeric_df.index = all_df.index
# combined_df = process_categorical(numeric_df, all_df, categorical_features)
# X = combined_df.as_matrix()
# X.shape

# #PCA
# from sklearn.decomposition import PCA

# test_n = df.shape[0]
# x = X[:test_n,:]

# pca = PCA()
# #Todo: Fit and transform X using PCA (function params: training data and labels)
# pca.fit(x,y)
# X = pca.transform(X)

# X.shape

# x = X[:test_n,:]
# x_test = X[test_n:,:]

# lr = linear_model.LinearRegression()
# lr.fit(x_train, y_train)

# ridge = linear_model.Ridge() 
# linear_model.Ridge.fit(x_train,y_train)
# ridge.predict(x_train)

# print('Linear Regression score is %f' % lr.score(x_val, y_val))
# print('Ridge score is %f' % ridge.score(x_val, y_val))


