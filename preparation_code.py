# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:06:45 2022

@author: leube
"""

import pandas as pd
import numpy as np

neo = pd.read_csv(r"C:\Users\leube\Downloads\neo.csv")
neo2 = pd.read_csv(r"C:\Users\leube\Downloads\neo.csv")


neo2['id'].value_counts()
filters = neo2[neo2['id']==2469219]

filcol = list(filters.columns)
ids = list(neo2['id'].unique())
for x in ids:
    filters = neo2[neo2['id']==x]
    if len(list(filters['hazardous'].unique()))>1:
        print(x, 'is a problem')
        
        

for x in ids:
    filters = neo2[neo2['id']==x]
    meanv = filters['relative_velocity'].mean()
    meand = filters['miss_distance'].mean()
    neo2.loc[neo2[neo2['id']==x].index, 'relative_velocity'] = meanv
    neo2.loc[neo2[neo2['id']==x].index, 'miss_distance'] = meand
    

for x in ids:
    filters = neo2[neo2['id']==x]
    if len(list(filters['relative_velocity'].unique()))>1:
        print(x, 'is a problem')
        
        
for x in ids:
    filters = neo2[neo2['id']==x]
    if len(list(filters['miss_distance'].unique()))>1:
        print(x, 'is a problem')
        
neonew = neo2.drop_duplicates(subset=['id'])


neonew.info()
neonew.shape

allcol = list(neonew.columns)
allcol
for x in allcol:
    print(x)
    print(neonew[x].unique())
    print('\n')

# all same values for column oribiting body and sentry object, will not have an
# impact on model decision if these values are the same for all asteriods --> drop the columns

del neonew['sentry_object']
del neonew['orbiting_body']


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 15))
heatmap = sns.heatmap(neonew.corr(), vmin=-1, vmax=1, annot=True)

# high correlation between 

del neonew['est_diameter_min']

neonew.describe()

numcol = list(neonew.describe().columns)

import seaborn as sns
import matplotlib.pyplot as plt

for x in numcol:
    sns.boxplot(data = neonew, y=x)
    plt.title(x)
    plt.show()

for x in numcol:
    Q3 = neonew[x].quantile(0.75)
    Q1= neonew[x].quantile(0.25)
    IQR = Q3 - Q1
    filters = neonew[neonew[x]>Q3 + 1.5*IQR]
    print(x)
    print(len(filters))
    print('\n')

neonew['hazardous'].value_counts()

del neonew['id']
del neonew['name']


neonew['hazardous'] = np.where(neonew['hazardous']==True,1,0)
allcol
for x in allcol:
    sns.displot(neonew, x=x)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def generate_results(prediction, y_test, model, x_test):
    print('The accuracy of the Tree is', '{:.3f}'.format(metrics.accuracy_score(prediction,y_test)))

    # matrix

    cm = metrics.plot_confusion_matrix(model, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
    cm.ax_.set_title('Tree Confusion matrix, without normalization');
 
    
 
    # feature selection for data including outliers as well
    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE # feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector


x = neonew.drop('hazardous', axis=1)
y = neonew['hazardous']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

def feature_selection_SFM(x,y):
    SFM = SelectFromModel(estimator=RandomForestClassifier()).fit(x, y)
    s=SFM.fit(x, y)
    return s.get_support()

def feature_selection_RFE(x,y):
    rfe_selector = RFE(estimator=RandomForestClassifier(),n_features_to_select = 3, step = 1)
    m=rfe_selector.fit(x, y)
    x.columns[m.get_support()]
    return print("Num Features: %s" % (m.n_features_), '\n', "Selected Features: %s" % (m.support_), '\n', "Feature Ranking: %s" % (m.ranking_))
                  
def feature_selection_RFECV(x,y):
    rfecv = RFECV(
        estimator=RandomForestClassifier(),
        min_features_to_select=3,
        step=4,
        n_jobs=-1,
        scoring="r2",
        cv=5,
    )
    
    m= rfecv.fit(x, y)
    m.ranking_
    return x.columns[rfecv.support_]

def feature_selection_SFS(x,y):
    sfs = SequentialFeatureSelector(estimator=RandomForestClassifier(),  n_features_to_select=3)
    m = sfs.fit(x,y)
    return x.columns[m.get_support()]

# SFM
feature_selection_SFM(x,y)  
  
allcol = list(neonew.columns)
allcol
mostimcol1 = allcol[2:3]

x_train2 = x_train[mostimcol1]
x_test2 = x_test[mostimcol1]

#RFE
feature_selection_RFE(x, y)

mostimcol2 = allcol[:3]

x_train3 = x_train[mostimcol2]
x_test3 = x_test[mostimcol2]

# RFECV
feature_selection_RFECV(x, y)

mostimcol3 = allcol[1:4]
mostimcol3
x_train4 = x_train[mostimcol3]
x_test4 = x_test[mostimcol3]

# SFS

feature_selection_SFS(x, y)
allcol
mostimcol4 = ['est_diameter_max', 'miss_distance', 'absolute_magnitude']
x_train5 = x_train[mostimcol4]
x_test5 = x_test[mostimcol4]

# due to timelimitation and the fact that we only have a limited amount of
# features, we will use our original traindata with all features only for the
# next steps of analysis

# hyperparameters decision


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def HS_Gridsearch(model,dicparam,score, xtrain, ytrain):
    grid_search = GridSearchCV(model,
                               dicparam,cv=5, scoring=score,verbose=1,n_jobs=-1
                               )
    grid_search.fit(xtrain,ytrain)
    grid_search.best_params_
    return print(grid_search.best_params_, grid_search.best_score_)
    
    

def HS_Randomsearch(model,dicparam,score, xtrain, ytrain):
    grid_search = GridSearchCV(model,
                               dicparam,cv=5, scoring=score,verbose=1,n_jobs=-1
                               )
    grid_search.fit(xtrain,ytrain)
    grid_search.best_params_
    return print(grid_search.best_params_, grid_search.best_score_)


d = {'n_estimators':np.arange(100,200,10),'max_features':['sqrt','log2',None],'min_samples_leaf':np.arange(1,5)}

# hyper paramter testing for forest train

HS_Randomsearch(RandomForestClassifier(), dicparam = d, score='f1', xtrain=x_train, ytrain=y_train)

dicparam = {'n_estimators':np.arange(100,200,10),
'max_features':['sqrt','log2',None],
'min_samples_leaf':np.arange(1,5)}

# result: {'max_features': None, 'min_samples_leaf': 1, 'n_estimators': 110} 0.37540589163911586
# very bad f1 score --> questionable if 

# hyper paramter testing for logistic regression
# our research has concluded that there is no necesity to tune the parameters
# of this model

# hyper parameter testing for NuSVC
from sklearn.svm import NuSVC
d={'nu':np.arange(0,1,0.1)}

HS_Randomsearch(NuSVC(), dicparam = d, score='f1', xtrain=x_train, ytrain=y_train)


# hyper parameter Bernoulli 
from sklearn.naive_bayes import BernoulliNB

d = {}

HS_Randomsearch(BernoulliNB(), dicparam = {'fit_prior':[True,False]}, score='f1', xtrain=x_train, ytrain=y_train)

# result: {'fit_prior': True} 0.0

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

HS_Randomsearch(AdaBoostClassifier(), dicparam = {'n_estimators':np.arange(100,200,10)}, score='f1', xtrain=x_train, ytrain=y_train)

# result: 160

# passive agressive
from sklearn.linear_model import PassiveAggressiveClassifier

HS_Randomsearch(PassiveAggressiveClassifier(), dicparam = {'fit_intercept':[True,False],'max_iter':[1000,2000,3000]}, score='f1', xtrain=x_train, ytrain=y_train)

# result: {'fit_intercept': True, 'max_iter': 2000} 0.023195266272189347

# SGDClassifier
from sklearn.linear_model import SGDClassifier

HS_Randomsearch(SGDClassifier(), dicparam = {'fit_intercept':[True,False],'max_iter':[1000,2000,3000]}, score='f1', xtrain=x_train, ytrain=y_train)

# result: {'fit_intercept': False, 'max_iter': 1000} 0.030345736920034155

# ComplementNB
from sklearn.naive_bayes import ComplementNB

HS_Randomsearch(ComplementNB(), dicparam = {'norm':[True,False], 'fit_prior':[True,False]}, score='f1', xtrain=x_train, ytrain=y_train)

# result: {'fit_prior': True, 'norm': False} 0.16874863841491952

np.arange(0,1,0.1)


# testing our models one by one
# adding roc auc information
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
model = LogisticRegression()
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()


# NuSVC
model = NuSVC(nu=0.2)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# BernoulliNB

model = BernoulliNB(fit_prior=True)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# lets try a different data set
model = BernoulliNB(fit_prior=True)
result = model.fit(x_train3,y_train)
prediction = result.predict(x_test3)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test3)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=160)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(fit_intercept= True,max_iter= 2000)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# SGDClassifier
model = SGDClassifier(fit_intercept= False,max_iter= 1000)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()


# ComplementNB
model = ComplementNB(norm=False, fit_prior=True)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()

# little test try on the side


from sklearn.gaussian_process import GaussianProcessClassifier

model = GaussianProcessClassifier(n_jobs=-1)
result = model.fit(x_train,y_train)
prediction = result.predict(x_test)

print(classification_report(y_test,prediction))
generate_results(prediction,y_test,model,x_test)

print(metrics.roc_auc_score(y_test, prediction))
metrics.plot_roc_curve(model, x_test, y_test)

plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()






    