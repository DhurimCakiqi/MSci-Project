import os
from parameters import parameters as pm,SklearnHelper
import pandas as pd
import numpy as np
import random
import time
import warnings
warnings.filterwarnings('ignore') # 忽略warnings

# Load  data from local file 
df=pd.read_excel(pm.file_datasets,index_col=0)
df.columns=['q1', 'q2', 'q3', 'q4', r'$\alpha_0$', r'$\beta_0$', r'$\alpha_1$',
       r'$\beta_1$', r'$\alpha_2$', r'$\beta_2$']
feature_columns=[ r'$\alpha_0$', r'$\beta_0$', r'$\alpha_1$',
       r'$\beta_1$', r'$\alpha_2$', r'$\beta_2$']
for col in feature_columns:
    df[col]=(df[col]-df[col].mean())/df[col].std()
    
q_data,y_data=df.iloc[:,:4],df.iloc[:,4:]

train_size=int(0.9*len(q_data))
q_data_test,y_data_test=q_data.iloc[train_size:,:].values,y_data.iloc[train_size:,:].values

def muilt_model_predict(x,model_tag="xgboost"):
    import joblib
    file_model=os.path.join(pm.abs_dir,'results',model_tag+'_model.m')
    models = joblib.load(file_model)
    y_pre=np.zeros(y_data_test.shape)
    for model in models:
        y_pre+=model.predict(x)/5
    return y_pre
    

XGBoost_y_pre=muilt_model_predict(q_data_test,model_tag="XGBoost")
## visualize
import matplotlib.pyplot as plt
title=[r'XGBoost-${\alpha}_0$', r'XGBoost-${\beta}_0$', r'XGBoost-${\alpha}_1$'
                    ,r'XGBoost-${\beta}_1$', r'XGBoost-${\alpha}_2$', r'XGBoost-${\beta}_2$']
     
for i in range(len(y_data_test[0])):
    plt.figure(figsize=(6.4, 4.8))
    plt.title(title[i],fontdict={'family' : 'Times New Roman'
                                       , 'size' : 28})
    y_true,y_pred=y_data_test[:,i],XGBoost_y_pre[:,i]                                   
    plt.scatter(y_true,y_pred,s=5,marker='o',alpha=0.8, )
    y_line=np.linspace(np.min(y_true),np.max(y_true),1000)
    plt.plot(y_line,y_line,'r-',linewidth=1)
    plt.yticks(fontproperties = 'Times New Roman', size = 28)
    plt.xticks(fontproperties = 'Times New Roman', size = 28)
    plt.xlabel('True value',fontdict={'family' : 'Times New Roman'
                                    , 'size' : 28})
    plt.ylabel('Predicted value',fontdict={'family' : 'Times New Roman'
                                    , 'size' : 28}) 
    print("MSE between true value and predicted value for {}-th feature:{}".
                format(i+1,np.mean(np.square(np.subtract(y_true,y_pred)))))
    del y_true,y_pred
plt.show()
