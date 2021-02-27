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
import os
from parameters import parameters as pm
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
#from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import timeit
import joblib

import warnings
warnings.filterwarnings('ignore')


df_1=pd.read_excel('../sampling/samples_para_stress_gamma_0.1.xlsx')  
df_2=pd.read_excel('../sampling/samples_para_stress_gamma_0.2.xlsx')
df_3=pd.read_excel('../sampling/samples_para_stress_gamma_0.3.xlsx') 
df_4=pd.read_excel('../sampling/samples_para_stress_gamma_0.4.xlsx')  
df_5=pd.read_excel('../sampling/samples_para_stress_gamma_0.5.xlsx')    

df_total=[df_1,df_2,df_3,df_4,df_5]


for i in df_total:
    i.columns=['a','b','af','bf','as','bs','afs','bfs','sigma_fs_fs','sigma_sf_fs','sigma_fn_fn','sigma_nf_fn','sigma_ns_sn','sigma_sn_sn']
    
 
feature_columns=['sigma_fs_fs','sigma_sf_fs','sigma_fn_fn','sigma_nf_fn','sigma_ns_sn','sigma_sn_sn']
parameters = ['a','b','af','bf','as','bs','afs','bfs']

feature_columns=['sigma_fs_fs','sigma_sf_fs','sigma_fn_fn','sigma_nf_fn','sigma_ns_sn','sigma_sn_sn']
parameters = ['a','b','af','bf','as','bs','afs','bfs']
col_mean = np.zeros((5,6))
col_std  = np.zeros((5,6))

#for i, df in enumerate(df_total):
#   for j, col in enumerate(feature_columns):
#        col_mean[i][j] = df[col].mean()
#        col_std[i][j] = df[col].std()
#        df[col]=(df[col]-df[col].mean())/df[col].std()        
        
# for saving this, your future test, the direct prediction is for the standardalized value 

q_1data=df_1.iloc[:,0:8].values
q_2data=df_2.iloc[:,0:8].values
q_3data=df_3.iloc[:,0:8].values
q_4data=df_4.iloc[:,0:8].values
q_5data=df_5.iloc[:,0:8].values

y_1data=df_1.iloc[:,8:].values
y_2data=df_2.iloc[:,8:].values
y_3data=df_3.iloc[:,8:].values
y_4data=df_4.iloc[:,8:].values
y_5data=df_5.iloc[:,8:].values
    

train_size=int(0.9*len(q_1data))


def mse_loss(y_true,y_pred):
    return np.mean(np.square(y_true-y_pred))

X_1_train = q_1data[:train_size,:]
y_1_train = y_1data[:train_size,:]
X_1_test = q_1data[train_size:,:]
y_1_test = y_1data[train_size:,:]

X_2_train = q_2data[:train_size,:]
y_2_train = y_2data[:train_size,:]
X_2_test = q_2data[train_size:,:]
y_2_test = y_2data[train_size:,:]

X_3_train = q_3data[:train_size,:]
y_3_train = y_3data[:train_size,:]
X_3_test = q_3data[train_size:,:]
y_3_test = y_3data[train_size:,:]

X_4_train = q_4data[:train_size,:]
y_4_train = y_4data[:train_size,:]
X_4_test = q_4data[train_size:,:]
y_4_test = y_4data[train_size:,:]
    
X_5_train = q_5data[:train_size,:]
y_5_train = y_5data[:train_size,:]
X_5_test = q_5data[train_size:,:]
y_5_test = y_5data[train_size:,:]

  

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern,RationalQuadratic

#gamma 0.1
start = timeit.default_timer()
kernel = ConstantKernel(100, (1e-3, 1e3))*RBF(length_scale= 1.0, length_scale_bounds=(0.0,1000))
gpr_1models=[]
for feat_flag in range(6):
    gpr_1sub = GaussianProcessRegressor(kernel=kernel,
                    random_state=0)
    print('fitting feature %d' % feat_flag)
    gpr_1sub.fit(X_1_train, y_1_train[:, feat_flag])
    print(gpr_1sub.kernel_)
    gpr_1models.append(gpr_1sub)
    
stop = timeit.default_timer()
print('Time: ', stop - start)

# gamma 0.2
start = timeit.default_timer()
kernel = ConstantKernel(100, (1e-3, 1e3))*RBF(length_scale= 1.0, length_scale_bounds=(0.0,100))
gpr_2models=[]
for feat_flag in range(6):
    gpr_2sub = GaussianProcessRegressor(kernel=kernel,
                    random_state=0)
    print('fitting feature %d' % feat_flag)
    gpr_2sub.fit(X_2_train, y_2_train[:, feat_flag])
    print(gpr_2sub.kernel_)
    gpr_2models.append(gpr_2sub)
stop = timeit.default_timer()
print('Time: ', stop - start)


#gamma 0.3
start = timeit.default_timer()
kernel = ConstantKernel(100, (1e-3, 1e3))*RBF(length_scale= 1.0, length_scale_bounds=(0.0,100))
gpr_3models=[]
for feat_flag in range(6):
    gpr_3sub = GaussianProcessRegressor(kernel=kernel,
                    random_state=0)
    print('fitting feature %d' % feat_flag)
    gpr_3sub.fit(X_3_train, y_3_train[:, feat_flag])
    print(gpr_3sub.kernel_)
    gpr_3models.append(gpr_3sub)
stop = timeit.default_timer()
print('Time: ', stop - start)


#gamma 0.4
start = timeit.default_timer()
kernel = ConstantKernel(100, (1e-3, 1e3))*RBF(length_scale= 1.0, length_scale_bounds=(0.0,100))
gpr_4models=[]
for feat_flag in range(6):
    gpr_4sub = GaussianProcessRegressor(kernel=kernel,
                    random_state=0)
    print('fitting feature %d' % feat_flag)
    gpr_4sub.fit(X_4_train, y_4_train[:, feat_flag])
    print(gpr_4sub.kernel_)
    gpr_4models.append(gpr_4sub)
stop = timeit.default_timer()
print('Time: ', stop - start)

#gamma 0.5
start = timeit.default_timer()
kernel = ConstantKernel(100, (1e-3, 1e3))*RBF(length_scale= 1.0, length_scale_bounds=(0.0,100))
gpr_5models=[]
for feat_flag in range(6):
    gpr_5sub = GaussianProcessRegressor(kernel=kernel,
                    random_state=0)
    print('fitting feature %d' % feat_flag)
    gpr_5sub.fit(X_5_train, y_5_train[:, feat_flag])
    print(gpr_5sub.kernel_)
    gpr_5models.append(gpr_5sub)
stop = timeit.default_timer()
print('Time: ', stop - start)

# gamma 0.1 gaussian processor test cases
for feat_flag in range(6):
    print('fitting feature %d' % feat_flag)
    gpr_1sub = gpr_1models[feat_flag]
    pred = gpr_1sub.predict(X_1_test)
    print('gpr Test: mse loss={:.6f} r2_score={:.6f}'.format(
        mse_loss(y_1_test[:, feat_flag], pred),metrics.r2_score(y_1_test[:, feat_flag], pred)))


# predict with a one set of paramter 
gamma_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(1,-1)
para_val = np.array([2,	5,	3,	6,	8,	7,	4,	8]).reshape(1,-1)


#predict for sigma_fs_fs
sigma_fs_fs = []
sigma_fs_fs.append(gpr_1models[0].predict(para_val) )
sigma_fs_fs.append(gpr_2models[0].predict(para_val) )
sigma_fs_fs.append(gpr_3models[0].predict(para_val) )
sigma_fs_fs.append(gpr_4models[0].predict(para_val) )
sigma_fs_fs.append(gpr_5models[0].predict(para_val) )

#predict for sigma_sf_fs
sigma_sf_fs = []
sigma_sf_fs.append(gpr_1models[1].predict(para_val) )
sigma_sf_fs.append(gpr_2models[1].predict(para_val) )
sigma_sf_fs.append(gpr_3models[1].predict(para_val) )
sigma_sf_fs.append(gpr_4models[1].predict(para_val) )
sigma_sf_fs.append(gpr_5models[1].predict(para_val) )

#predict for sigma_fn_fn
sigma_fn_fn = []
sigma_fn_fn.append(gpr_1models[2].predict(para_val) )
sigma_fn_fn.append(gpr_2models[2].predict(para_val) )
sigma_fn_fn.append(gpr_3models[2].predict(para_val) )
sigma_fn_fn.append(gpr_4models[2].predict(para_val) )
sigma_fn_fn.append(gpr_5models[2].predict(para_val) )

#predict for sigma_nf_fn
sigma_nf_fn = []
sigma_nf_fn.append(gpr_1models[3].predict(para_val) )
sigma_nf_fn.append(gpr_2models[3].predict(para_val) )
sigma_nf_fn.append(gpr_3models[3].predict(para_val) )
sigma_nf_fn.append(gpr_4models[3].predict(para_val) )
sigma_nf_fn.append(gpr_5models[3].predict(para_val) )

#predict for sigma_ns_sn
sigma_ns_sn = []
sigma_ns_sn.append(gpr_1models[4].predict(para_val) )
sigma_ns_sn.append(gpr_2models[4].predict(para_val) )
sigma_ns_sn.append(gpr_3models[4].predict(para_val) )
sigma_ns_sn.append(gpr_4models[4].predict(para_val) )
sigma_ns_sn.append(gpr_5models[4].predict(para_val) )

#predict for sigma_sn_sn
sigma_sn_sn = []
sigma_sn_sn.append(gpr_1models[5].predict(para_val) )
sigma_sn_sn.append(gpr_2models[5].predict(para_val) )
sigma_sn_sn.append(gpr_3models[5].predict(para_val) )
sigma_sn_sn.append(gpr_4models[5].predict(para_val) )
sigma_sn_sn.append(gpr_5models[5].predict(para_val) )


# load the analytical solution 
df_test_analy=pd.read_excel('../sampling/test_stress_variedGamma.xlsx');
sigma_fs_fs_analy = df_test_analy.iloc[:,0].values;
sigma_sf_fs_analy = df_test_analy.iloc[:,1].values;
sigma_fn_fn_analy = df_test_analy.iloc[:,2].values;
sigma_nf_fn_analy = df_test_analy.iloc[:,3].values;
sigma_ns_sn_analy = df_test_analy.iloc[:,4].values;
sigma_sn_sn_analy = df_test_analy.iloc[:,5].values;
gamma_analy       = df_test_analy.iloc[:,6].values;


fig, axs = plt.subplots(3, 2, figsize=(8,8))
axs[0, 0].plot(gamma_analy, sigma_fs_fs_analy, linestyle='dashed', label=r'$\sigma_{fs}^{fs}$')
axs[0, 0].scatter(gamma_val, sigma_fs_fs,c='red', label='prediction')
axs[0, 0].set_xlabel(r'$\gamma$')
axs[0, 0].set_ylabel(r'$\sigma_{fs}^{fs}$')

axs[0, 1].plot(gamma_analy, sigma_sf_fs_analy, linestyle='dashed', label=r'$\sigma_{fs}^{fs}$')
axs[0, 1].scatter(gamma_val, sigma_sf_fs,c='red', label='prediction')
axs[0, 1].set_ylabel(r'$\sigma_{sf}^{fs}$')
axs[0, 1].set_xlabel(r'$\gamma$')

axs[1, 0].plot(gamma_analy, sigma_fn_fn_analy, linestyle='dashed', label=r'$\sigma_{fs}^{fs}$')
axs[1, 0].scatter(gamma_val, sigma_fn_fn,c='red', label='prediction')
axs[1, 0].set_ylabel(r'$\sigma_{fn}^{fn}$')
axs[1, 0].set_xlabel(r'$\gamma$')

axs[1, 1].plot(gamma_analy, sigma_nf_fn_analy, linestyle='dashed', label=r'$\sigma_{fs}^{fs}$')
axs[1, 1].scatter(gamma_val, sigma_nf_fn,c='red', label='prediction')
axs[1, 1].set_ylabel(r'$\sigma_{nf}^{fn}$')
axs[1, 1].set_xlabel(r'$\gamma$')

axs[2, 0].plot(gamma_analy, sigma_ns_sn_analy, linestyle='dashed', label=r'$\sigma_{fs}^{fs}$')
axs[2, 0].scatter(gamma_val, sigma_ns_sn,c='red', label='prediction')
axs[2, 0].set_ylabel(r'$\sigma_{ns}^{sn}$')
axs[2, 0].set_xlabel(r'$\gamma$')

axs[2, 1].plot(gamma_analy, sigma_sn_sn_analy, linestyle='dashed', label=r'$\sigma_{fs}^{fs}$')
axs[2, 1].scatter(gamma_val, sigma_sn_sn,c='red', label='prediction')
axs[2, 1].set_ylabel(r'$\sigma_{sn}^{sn}$')
axs[2, 1].set_xlabel(r'$\gamma$')

ax.legend()
fig.tight_layout()
plt.show()



    


    