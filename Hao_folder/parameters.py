# -*- coding: utf-8 -*-
import os
class parameters(object):
    numPoint=195
    abs_dir=os.getcwd()
    total_size=10000
    train_size=round(0.9*total_size)
    test_size=total_size-train_size

    strain_tags=['max','mid','min']
    use_tags=['max','min']
    file_presvol_exp=os.path.join(abs_dir,'dataSets','presVol','presvol.dat')
    file_strains_exp={}
    for tag in strain_tags:
        file_strains_exp[tag]=os.path.join(abs_dir,'dataSets','strain','strain'+tag,"interchangestrain"+tag+".dat")

    file_samples=os.path.join(abs_dir,'dataSets','samples.xls')
    file_datasets=os.path.join(abs_dir,'datasets.xlsx')
    file_mat=os.path.join(abs_dir,"dataSets","lsq_fun.mat")
    
# Class to extend the Sklearn regressor
class SklearnHelper(object):
    def __init__(self, clf, params=None,isMuilt_reg=False):
        if isMuilt_reg: self.clf= MultiOutputRegressor (clf(**params))
        else: self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def score(self,x,y):
        return self.clf.score(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


     



