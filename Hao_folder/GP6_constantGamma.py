# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
import os
from parameters import parameters as pm
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
#from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

import joblib

import warnings
warnings.filterwarnings('ignore')


df=pd.read_excel('samples_para_stress_nogamma.xlsx')
df.columns=['a','b','af','bf','as','bs','afs','bfs','sigma_fs_fs','sigma_sf_fs','sigma_fn_fn','sigma_nf_fn','sigma_ns_sn','sigma_sn_sn']

df.describe()

df.head(5)

feature_columns=['sigma_fs_fs','sigma_sf_fs','sigma_fn_fn','sigma_nf_fn','sigma_ns_sn','sigma_sn_sn']
for col in feature_columns:
    df[col]=(df[col]-df[col].mean())/df[col].std()
    

df[feature_columns].boxplot()
plt.gca().set_ylabel("value",fontdict={"size":18})
plt.gca().set_xlabel("feature name",fontdict={'family' : 'Times New Roman', 'size' : 18})
plt.show()


q_data,y_data=df.iloc[:,0:8],df.iloc[:,8:]

train_size=int(0.9*len(q_data))

q_data_train,q_data_test=q_data.iloc[:train_size,:],q_data.iloc[train_size:,:]
y_data_train,y_data_test=y_data.iloc[:train_size,:],y_data.iloc[train_size:,:]

#q_data_train.shape,q_data_test.shape

def mse_loss(y_true,y_pred):
    return np.mean(np.square(y_true-y_pred))

X = q_data_train.values
y = y_data_train.values
X_test = q_data_test.values
y_test=y_data_test.values

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(1.0, (1e-3, 1e3))*RBF(length_scale= 1.0, length_scale_bounds=(0.0,100))
# kernel = RBF(**gpr_para[feat_flag])

gpr_models=[]
for feat_flag in  range(len(feature_columns)):
    gpr_sub = GaussianProcessRegressor(kernel=kernel,
                         random_state=0)
    print('fitting feature %d' % feat_flag)
    gpr_sub.fit(X, y[:,feat_flag])
    print(gpr_sub.kernel_)
    gpr_models.append(gpr_sub)

y_test_pred = np.zeros(y_test.shape)
for ix, model in enumerate(gpr_models):
    y_test_pred[:,ix] += model.predict(X_test)
    print('gpr Test: mse loss={:.6f} r2_score={:.6f}'.format(
            mse_loss(y_test[:, ix], y_test_pred[:, ix]),metrics.r2_score(y_test[:,ix], y_test_pred[:, ix])))    

plt.style.use('seaborn-whitegrid')
plt.subplot(2,3,1)
feat_flag = 0;
feature_min = min(y_test_pred[:,feat_flag])
feature_max = max(y_test_pred[:,feat_flag])
feature_linspace = np.linspace(feature_min, feature_max, 100)
plt.plot(feature_linspace, feature_linspace, '-k', linewidth=4)
plt.scatter(y_test_pred[:,feat_flag], y_test[:,feat_flag], marker='o');
plt.xlabel('prediction')
plt.ylabel('ground truth');
plt.title('feature 1')
#plt.show()

plt.subplot(2,3,2)
feat_flag = 1;
feature_min = min(y_test_pred[:,feat_flag])
feature_max = max(y_test_pred[:,feat_flag])
feature_linspace = np.linspace(feature_min, feature_max, 100)
plt.plot(feature_linspace, feature_linspace, '-k', linewidth=4)
plt.scatter(y_test_pred[:,feat_flag], y_test[:,feat_flag], marker='o');
plt.xlabel('prediction')
plt.ylabel('ground truth');
plt.title('feature 2')
#plt.show()

plt.subplot(2,3,3)
feat_flag = 2;
feature_min = min(y_test_pred[:,feat_flag])
feature_max = max(y_test_pred[:,feat_flag])
feature_linspace = np.linspace(feature_min, feature_max, 100)
plt.plot(feature_linspace, feature_linspace, '-k', linewidth=4)
plt.scatter(y_test_pred[:,feat_flag], y_test[:,feat_flag], marker='o');
plt.xlabel('prediction')
plt.ylabel('ground truth');
plt.title('feature 3')
#plt.show()

plt.subplot(2,3,4)
feat_flag = 3;
feature_min = min(y_test_pred[:,feat_flag])
feature_max = max(y_test_pred[:,feat_flag])
feature_linspace = np.linspace(feature_min, feature_max, 100)
plt.plot(feature_linspace, feature_linspace, '-k', linewidth=4)
plt.scatter(y_test_pred[:,feat_flag], y_test[:,feat_flag], marker='o');
plt.xlabel('prediction')
plt.ylabel('ground truth');
plt.title('feature 4')
#plt.show()


plt.subplot(2,3,5)
feat_flag = 4;
feature_min = min(y_test_pred[:,feat_flag])
feature_max = max(y_test_pred[:,feat_flag])
feature_linspace = np.linspace(feature_min, feature_max, 100)
plt.plot(feature_linspace, feature_linspace, '-k', linewidth=4)
plt.scatter(y_test_pred[:,feat_flag], y_test[:,feat_flag], marker='o');
plt.xlabel('prediction')
plt.ylabel('ground truth');
plt.title('feature 4')
#plt.show()

plt.subplot(2,3,6)
feat_flag = 5;
feature_min = min(y_test_pred[:,feat_flag])
feature_max = max(y_test_pred[:,feat_flag])
feature_linspace = np.linspace(feature_min, feature_max, 100)
plt.plot(feature_linspace, feature_linspace, '-k', linewidth=4)
plt.scatter(y_test_pred[:,feat_flag], y_test[:,feat_flag], marker='o');
plt.xlabel('prediction')
plt.ylabel('ground truth');
plt.title('feature 6')
plt.show()

















