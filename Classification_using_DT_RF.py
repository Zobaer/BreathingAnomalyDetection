

import numpy as np
import pandas as pd
#import graphviz
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


data_half = np.genfromtxt('hf_half.csv',delimiter=',')
x_data_half = data_half[:,1:-1]
y_data_half = data_half[:,-1]
#Split in train-validation-test sets 60-20-20
#a-b-c% => x = 100b/(100-c)
x_tv_half, x_test_half, y_tv_half, y_test_half = train_test_split(x_data_half, y_data_half, test_size=0.2, shuffle = True, random_state=1)
x_train_half, x_val_half, y_train_half, y_val_half = train_test_split(x_tv_half, y_tv_half, test_size=0.25, shuffle = True, random_state=1)

data_one = np.genfromtxt('hf_one.csv',delimiter=',')
x_data_one = data_one[:,1:-1]
y_data_one = data_one[:,-1]
#Split in train-validation-test sets 60-20-20
#a-b-c% => x = 100b/(100-c)
x_tv_one, x_test_one, y_tv_one, y_test_one = train_test_split(x_data_one, y_data_one, test_size=0.2, shuffle = True, random_state=1)
x_train_one, x_val_one, y_train_one, y_val_one = train_test_split(x_tv_one, y_tv_one, test_size=0.25, shuffle = True, random_state=1)

data_oneandhalf = np.genfromtxt('hf_oneandhalf.csv',delimiter=',')
x_data_oneandhalf = data_oneandhalf[:,1:-1]
y_data_oneandhalf = data_oneandhalf[:,-1]
#Split in train-validation-test sets 60-20-20
#a-b-c% => x = 100b/(100-c)
x_tv_oneandhalf, x_test_oneandhalf, y_tv_oneandhalf, y_test_oneandhalf = train_test_split(x_data_oneandhalf, y_data_oneandhalf, test_size=0.2, shuffle = True, random_state=1)
x_train_oneandhalf, x_val_oneandhalf, y_train_oneandhalf, y_val_oneandhalf = train_test_split(x_tv_oneandhalf, y_tv_oneandhalf, test_size=0.25, shuffle = True, random_state=1)

data_half_one = np.genfromtxt('hf_half_one.csv',delimiter=',')
x_data_half_one = data_half_one[:,1:-1]
y_data_half_one = data_half_one[:,-1]
#Split in train-validation-test sets 60-20-20
#a-b-c% => x = 100b/(100-c)
x_tv_half_one, x_test_half_one, y_tv_half_one, y_test_half_one = train_test_split(x_data_half_one, y_data_half_one, test_size=0.2, shuffle = True, random_state=1)
x_train_half_one, x_val_half_one, y_train_half_one, y_val_half_one = train_test_split(x_tv_half_one, y_tv_half_one, test_size=0.25, shuffle = True, random_state=1)

data_one_oneandhalf = np.genfromtxt('hf_one_oneandhalf.csv',delimiter=',')
x_data_one_oneandhalf = data_one_oneandhalf[:,1:-1]
y_data_one_oneandhalf = data_one_oneandhalf[:,-1]
#Split in train-validation-test sets 60-20-20
#a-b-c% => x = 100b/(100-c)
x_tv_one_oneandhalf, x_test_one_oneandhalf, y_tv_one_oneandhalf, y_test_one_oneandhalf = train_test_split(x_data_one_oneandhalf, y_data_one_oneandhalf, test_size=0.2, shuffle = True, random_state=1)
x_train_one_oneandhalf, x_val_one_oneandhalf, y_train_one_oneandhalf, y_val_one_oneandhalf = train_test_split(x_tv_one_oneandhalf, y_tv_one_oneandhalf, test_size=0.25, shuffle = True, random_state=1)

data_all = np.genfromtxt('hf_navg.csv',delimiter=',')
x_data_all = data_all[:,1:-1]
y_data_all = data_all[:,-1]
#Split in train-validation-test sets 60-20-20
#a-b-c% => x = 100b/(100-c)
x_tv_all, x_test_all, y_tv_all, y_test_all = train_test_split(x_data_all, y_data_all, test_size=0.2, shuffle = True, random_state=1)
x_train_all, x_val_all, y_train_all, y_val_all = train_test_split(x_tv_all, y_tv_all, test_size=0.25, shuffle = True, random_state=1)

#print("Shape of training data: %s" % (x_train.shape,))
#print("Shape of validation data: %s" % (x_val.shape,))
#print("Shape of test data: %s" % (x_test.shape,))

param1_start = 1
param1_end = 25

#accuracies will be stored here
dt_train_accuracy_half = np.zeros((param1_end-param1_start+1))
dt_val_accuracy_half = np.zeros((param1_end-param1_start+1))
rf_train_accuracy_half = np.zeros((param1_end-param1_start+1))
rf_val_accuracy_half = np.zeros((param1_end-param1_start+1))

dt_train_accuracy_one = np.zeros((param1_end-param1_start+1))
dt_val_accuracy_one = np.zeros((param1_end-param1_start+1))
rf_train_accuracy_one = np.zeros((param1_end-param1_start+1))
rf_val_accuracy_one = np.zeros((param1_end-param1_start+1))

dt_train_accuracy_oneandhalf = np.zeros((param1_end-param1_start+1))
dt_val_accuracy_oneandhalf = np.zeros((param1_end-param1_start+1))
rf_train_accuracy_oneandhalf = np.zeros((param1_end-param1_start+1))
rf_val_accuracy_oneandhalf = np.zeros((param1_end-param1_start+1))

dt_train_accuracy_half_one = np.zeros((param1_end-param1_start+1))
dt_val_accuracy_half_one = np.zeros((param1_end-param1_start+1))
rf_train_accuracy_half_one = np.zeros((param1_end-param1_start+1))
rf_val_accuracy_half_one = np.zeros((param1_end-param1_start+1))

dt_train_accuracy_one_oneandhalf = np.zeros((param1_end-param1_start+1))
dt_val_accuracy_one_oneandhalf = np.zeros((param1_end-param1_start+1))
rf_train_accuracy_one_oneandhalf = np.zeros((param1_end-param1_start+1))
rf_val_accuracy_one_oneandhalf = np.zeros((param1_end-param1_start+1))

dt_train_accuracy_all = np.zeros((param1_end-param1_start+1))
dt_val_accuracy_all = np.zeros((param1_end-param1_start+1))
rf_train_accuracy_all = np.zeros((param1_end-param1_start+1))
rf_val_accuracy_all = np.zeros((param1_end-param1_start+1))

for i in range(param1_start,param1_end+1):
    dt_model = DecisionTreeClassifier(max_depth = i) #max_depth = 10

    dt_model.fit(x_train_half, y_train_half)
    dt_y_train_pred_half = dt_model.predict(x_train_half)
    dt_y_val_pred_half = dt_model.predict(x_val_half)
    dt_train_accuracy_half[i-param1_start] = accuracy_score(y_true=y_train_half, y_pred=dt_y_train_pred_half)
    dt_val_accuracy_half[i-param1_start]= accuracy_score(y_true=y_val_half, y_pred=dt_y_val_pred_half)

    dt_model.fit(x_train_one, y_train_one)
    dt_y_train_pred_one = dt_model.predict(x_train_one)
    dt_y_val_pred_one = dt_model.predict(x_val_one)
    dt_train_accuracy_one[i-param1_start] = accuracy_score(y_true=y_train_one, y_pred=dt_y_train_pred_one)
    dt_val_accuracy_one[i-param1_start]= accuracy_score(y_true=y_val_one, y_pred=dt_y_val_pred_one)

    dt_model.fit(x_train_oneandhalf, y_train_oneandhalf)
    dt_y_train_pred_oneandhalf = dt_model.predict(x_train_oneandhalf)
    dt_y_val_pred_oneandhalf = dt_model.predict(x_val_oneandhalf)
    dt_train_accuracy_oneandhalf[i-param1_start] = accuracy_score(y_true=y_train_oneandhalf, y_pred=dt_y_train_pred_oneandhalf)
    dt_val_accuracy_oneandhalf[i-param1_start]= accuracy_score(y_true=y_val_oneandhalf, y_pred=dt_y_val_pred_oneandhalf)

    dt_model.fit(x_train_half_one, y_train_half_one)
    dt_y_train_pred_half_one = dt_model.predict(x_train_half_one)
    dt_y_val_pred_half_one = dt_model.predict(x_val_half_one)
    dt_train_accuracy_half_one[i-param1_start] = accuracy_score(y_true=y_train_half_one, y_pred=dt_y_train_pred_half_one)
    dt_val_accuracy_half_one[i-param1_start]= accuracy_score(y_true=y_val_half_one, y_pred=dt_y_val_pred_half_one)


    dt_model.fit(x_train_one_oneandhalf, y_train_one_oneandhalf)
    dt_y_train_pred_one_oneandhalf = dt_model.predict(x_train_one_oneandhalf)
    dt_y_val_pred_one_oneandhalf = dt_model.predict(x_val_one_oneandhalf)
    dt_train_accuracy_one_oneandhalf[i-param1_start] = accuracy_score(y_true=y_train_one_oneandhalf, y_pred=dt_y_train_pred_one_oneandhalf)
    dt_val_accuracy_one_oneandhalf[i-param1_start]= accuracy_score(y_true=y_val_one_oneandhalf, y_pred=dt_y_val_pred_one_oneandhalf)

    dt_model.fit(x_train_all, y_train_all)
    dt_y_train_pred_all = dt_model.predict(x_train_all)
    dt_y_val_pred_all = dt_model.predict(x_val_all)
    dt_train_accuracy_all[i-param1_start] = accuracy_score(y_true=y_train_all, y_pred=dt_y_train_pred_all)
    dt_val_accuracy_all[i-param1_start]= accuracy_score(y_true=y_val_all, y_pred=dt_y_val_pred_all)

    rf_model = RandomForestClassifier(n_estimators=i, criterion='entropy',random_state=0)

    rf_model.fit(x_train_half, y_train_half)
    rf_y_train_pred_half = rf_model.predict(x_train_half)
    rf_y_val_pred_half = rf_model.predict(x_val_half)
    rf_train_accuracy_half[i-param1_start] = accuracy_score(y_true=y_train_half, y_pred=rf_y_train_pred_half)
    rf_val_accuracy_half[i-param1_start]= accuracy_score(y_true=y_val_half, y_pred=rf_y_val_pred_half)

    rf_model.fit(x_train_one, y_train_one)
    rf_y_train_pred_one = rf_model.predict(x_train_one)
    rf_y_val_pred_one = rf_model.predict(x_val_one)
    rf_train_accuracy_one[i-param1_start] = accuracy_score(y_true=y_train_one, y_pred=rf_y_train_pred_one)
    rf_val_accuracy_one[i-param1_start]= accuracy_score(y_true=y_val_one, y_pred=rf_y_val_pred_one)

    rf_model.fit(x_train_oneandhalf, y_train_oneandhalf)
    rf_y_train_pred_oneandhalf = rf_model.predict(x_train_oneandhalf)
    rf_y_val_pred_oneandhalf = rf_model.predict(x_val_oneandhalf)
    rf_train_accuracy_oneandhalf[i-param1_start] = accuracy_score(y_true=y_train_oneandhalf, y_pred=rf_y_train_pred_oneandhalf)
    rf_val_accuracy_oneandhalf[i-param1_start]= accuracy_score(y_true=y_val_oneandhalf, y_pred=rf_y_val_pred_oneandhalf)

    rf_model.fit(x_train_half_one, y_train_half_one)
    rf_y_train_pred_half_one = rf_model.predict(x_train_half_one)
    rf_y_val_pred_half_one = rf_model.predict(x_val_half_one)
    rf_train_accuracy_half_one[i-param1_start] = accuracy_score(y_true=y_train_half_one, y_pred=rf_y_train_pred_half_one)
    rf_val_accuracy_half_one[i-param1_start]= accuracy_score(y_true=y_val_half_one, y_pred=rf_y_val_pred_half_one)

    rf_model.fit(x_train_one_oneandhalf, y_train_one_oneandhalf)
    rf_y_train_pred_one_oneandhalf = rf_model.predict(x_train_one_oneandhalf)
    rf_y_val_pred_one_oneandhalf = rf_model.predict(x_val_one_oneandhalf)
    rf_train_accuracy_one_oneandhalf[i-param1_start] = accuracy_score(y_true=y_train_one_oneandhalf, y_pred=rf_y_train_pred_one_oneandhalf)
    rf_val_accuracy_one_oneandhalf[i-param1_start]= accuracy_score(y_true=y_val_one_oneandhalf, y_pred=rf_y_val_pred_one_oneandhalf)

    rf_model.fit(x_train_all, y_train_all)
    rf_y_train_pred_all = rf_model.predict(x_train_all)
    rf_y_val_pred_all = rf_model.predict(x_val_all)
    rf_train_accuracy_all[i-param1_start] = accuracy_score(y_true=y_train_all, y_pred=rf_y_train_pred_all)
    rf_val_accuracy_all[i-param1_start]= accuracy_score(y_true=y_val_all, y_pred=rf_y_val_pred_all)




#test accuracies
dt_model = DecisionTreeClassifier(max_depth = 10) #max_depth = 10
dt_model.fit(x_train_half, y_train_half)
dt_y_train_pred_half = dt_model.predict(x_train_half)
dt_y_val_pred_half = dt_model.predict(x_val_half)
dt_y_test_pred_half = dt_model.predict(x_test_half)
dt_train_accuracy_half2 = accuracy_score(y_true=y_train_half, y_pred=dt_y_train_pred_half)
dt_val_accuracy_half2= accuracy_score(y_true=y_val_half, y_pred=dt_y_val_pred_half)
dt_test_accuracy_half2= accuracy_score(y_true=y_test_half, y_pred=dt_y_test_pred_half)



dt_model.fit(x_train_one, y_train_one)
dt_y_train_pred_one = dt_model.predict(x_train_one)
dt_y_val_pred_one = dt_model.predict(x_val_one)
dt_y_test_pred_one = dt_model.predict(x_test_one)
dt_train_accuracy_one2 = accuracy_score(y_true=y_train_one, y_pred=dt_y_train_pred_one)
dt_val_accuracy_one2= accuracy_score(y_true=y_val_one, y_pred=dt_y_val_pred_one)
dt_test_accuracy_one2= accuracy_score(y_true=y_test_one, y_pred=dt_y_test_pred_one)

dt_model.fit(x_train_oneandhalf, y_train_oneandhalf)
dt_y_train_pred_oneandhalf = dt_model.predict(x_train_oneandhalf)
dt_y_val_pred_oneandhalf = dt_model.predict(x_val_oneandhalf)
dt_y_test_pred_oneandhalf = dt_model.predict(x_test_oneandhalf)
dt_train_accuracy_oneandhalf2 = accuracy_score(y_true=y_train_oneandhalf, y_pred=dt_y_train_pred_oneandhalf)
dt_val_accuracy_oneandhalf2= accuracy_score(y_true=y_val_oneandhalf, y_pred=dt_y_val_pred_oneandhalf)
dt_test_accuracy_oneandhalf2= accuracy_score(y_true=y_test_oneandhalf, y_pred=dt_y_test_pred_oneandhalf)

dt_model.fit(x_train_half_one, y_train_half_one)
dt_y_train_pred_half_one = dt_model.predict(x_train_half_one)
dt_y_val_pred_half_one = dt_model.predict(x_val_half_one)
dt_y_test_pred_half_one = dt_model.predict(x_test_half_one)
dt_train_accuracy_half_one2 = accuracy_score(y_true=y_train_half_one, y_pred=dt_y_train_pred_half_one)
dt_val_accuracy_half_one2= accuracy_score(y_true=y_val_half_one, y_pred=dt_y_val_pred_half_one)
dt_test_accuracy_half_one2= accuracy_score(y_true=y_test_half_one, y_pred=dt_y_test_pred_half_one)


dt_model.fit(x_train_one_oneandhalf, y_train_one_oneandhalf)
dt_y_train_pred_one_oneandhalf = dt_model.predict(x_train_one_oneandhalf)
dt_y_val_pred_one_oneandhalf = dt_model.predict(x_val_one_oneandhalf)
dt_y_test_pred_one_oneandhalf = dt_model.predict(x_test_one_oneandhalf)
dt_train_accuracy_one_oneandhalf2 = accuracy_score(y_true=y_train_one_oneandhalf, y_pred=dt_y_train_pred_one_oneandhalf)
dt_val_accuracy_one_oneandhalf2= accuracy_score(y_true=y_val_one_oneandhalf, y_pred=dt_y_val_pred_one_oneandhalf)
dt_test_accuracy_one_oneandhalf2= accuracy_score(y_true=y_test_one_oneandhalf, y_pred=dt_y_test_pred_one_oneandhalf)

dt_model.fit(x_train_all, y_train_all)
dt_y_train_pred_all = dt_model.predict(x_train_all)
dt_y_val_pred_all = dt_model.predict(x_val_all)
dt_y_test_pred_all = dt_model.predict(x_test_all)
dt_train_accuracy_all2 = accuracy_score(y_true=y_train_all, y_pred=dt_y_train_pred_all)
dt_val_accuracy_all2= accuracy_score(y_true=y_val_all, y_pred=dt_y_val_pred_all)
dt_test_accuracy_all2= accuracy_score(y_true=y_test_all, y_pred=dt_y_test_pred_all)



rf_model = RandomForestClassifier(n_estimators=12, criterion='entropy',random_state=0)

rf_model.fit(x_train_half, y_train_half)
rf_y_train_pred_half = rf_model.predict(x_train_half)
rf_y_val_pred_half = rf_model.predict(x_val_half)
rf_y_test_pred_half = rf_model.predict(x_test_half)
rf_train_accuracy_half2 = accuracy_score(y_true=y_train_half, y_pred=rf_y_train_pred_half)
rf_val_accuracy_half2= accuracy_score(y_true=y_val_half, y_pred=rf_y_val_pred_half)
rf_test_accuracy_half2= accuracy_score(y_true=y_test_half, y_pred=rf_y_test_pred_half)

rf_model.fit(x_train_one, y_train_one)
rf_y_train_pred_one = rf_model.predict(x_train_one)
rf_y_val_pred_one = rf_model.predict(x_val_one)
rf_y_test_pred_one = rf_model.predict(x_test_one)
rf_train_accuracy_one2 = accuracy_score(y_true=y_train_one, y_pred=rf_y_train_pred_one)
rf_val_accuracy_one2= accuracy_score(y_true=y_val_one, y_pred=rf_y_val_pred_one)
rf_test_accuracy_one2= accuracy_score(y_true=y_test_one, y_pred=rf_y_test_pred_one)

rf_model.fit(x_train_oneandhalf, y_train_oneandhalf)
rf_y_train_pred_oneandhalf = rf_model.predict(x_train_oneandhalf)
rf_y_val_pred_oneandhalf = rf_model.predict(x_val_oneandhalf)
rf_y_test_pred_oneandhalf = rf_model.predict(x_test_oneandhalf)
rf_train_accuracy_oneandhalf2 = accuracy_score(y_true=y_train_oneandhalf, y_pred=rf_y_train_pred_oneandhalf)
rf_val_accuracy_oneandhalf2= accuracy_score(y_true=y_val_oneandhalf, y_pred=rf_y_val_pred_oneandhalf)
rf_test_accuracy_oneandhalf2= accuracy_score(y_true=y_test_oneandhalf, y_pred=rf_y_test_pred_oneandhalf)

rf_model.fit(x_train_half_one, y_train_half_one)
rf_y_train_pred_half_one = rf_model.predict(x_train_half_one)
rf_y_val_pred_half_one = rf_model.predict(x_val_half_one)
rf_y_test_pred_half_one = rf_model.predict(x_test_half_one)
rf_train_accuracy_half_one2 = accuracy_score(y_true=y_train_half_one, y_pred=rf_y_train_pred_half_one)
rf_val_accuracy_half_one2= accuracy_score(y_true=y_val_half_one, y_pred=rf_y_val_pred_half_one)
rf_test_accuracy_half_one2= accuracy_score(y_true=y_test_half_one, y_pred=rf_y_test_pred_half_one)


rf_model.fit(x_train_one_oneandhalf, y_train_one_oneandhalf)
rf_y_train_pred_one_oneandhalf = rf_model.predict(x_train_one_oneandhalf)
rf_y_val_pred_one_oneandhalf = rf_model.predict(x_val_one_oneandhalf)
rf_y_test_pred_one_oneandhalf = rf_model.predict(x_test_one_oneandhalf)
rf_train_accuracy_one_oneandhalf2 = accuracy_score(y_true=y_train_one_oneandhalf, y_pred=rf_y_train_pred_one_oneandhalf)
rf_val_accuracy_one_oneandhalf2= accuracy_score(y_true=y_val_one_oneandhalf, y_pred=rf_y_val_pred_one_oneandhalf)
rf_test_accuracy_one_oneandhalf2= accuracy_score(y_true=y_test_one_oneandhalf, y_pred=rf_y_test_pred_one_oneandhalf)

rf_model.fit(x_train_all, y_train_all)
rf_y_train_pred_all = rf_model.predict(x_train_all)
rf_y_val_pred_all = rf_model.predict(x_val_all)
rf_y_test_pred_all = rf_model.predict(x_test_all)
rf_train_accuracy_all2 = accuracy_score(y_true=y_train_all, y_pred=rf_y_train_pred_all)
rf_val_accuracy_all2= accuracy_score(y_true=y_val_all, y_pred=rf_y_val_pred_all)
rf_test_accuracy_all2= accuracy_score(y_true=y_test_all, y_pred=rf_y_test_pred_all)


cv_model = KFold(n_splits = 10, random_state = 1, shuffle = True)
dt_cv_accuracy_list_half = cross_validate(dt_model,x_tv_half, y_tv_half, cv = cv_model, n_jobs = -1, return_train_score = True)
dt_cv_train_accuracy_half = np.mean(dt_cv_accuracy_list_half['train_score'])
dt_cv_test_accuracy_half = np.mean(dt_cv_accuracy_list_half['test_score'])
dt_cv_y_pred_half = cross_val_predict(dt_model,x_tv_half, y_tv_half, cv = cv_model, n_jobs = -1)


dt_cv_accuracy_list_one = cross_validate(dt_model,x_tv_one, y_tv_one, cv = cv_model, n_jobs = -1, return_train_score = True)
dt_cv_train_accuracy_one = np.mean(dt_cv_accuracy_list_one['train_score'])
dt_cv_test_accuracy_one = np.mean(dt_cv_accuracy_list_one['test_score'])
dt_cv_y_pred_one = cross_val_predict(dt_model,x_tv_one, y_tv_one, cv = cv_model, n_jobs = -1)

dt_cv_accuracy_list_oneandhalf = cross_validate(dt_model,x_tv_oneandhalf, y_tv_oneandhalf, cv = cv_model, n_jobs = -1, return_train_score = True)
dt_cv_train_accuracy_oneandhalf = np.mean(dt_cv_accuracy_list_oneandhalf['train_score'])
dt_cv_test_accuracy_oneandhalf = np.mean(dt_cv_accuracy_list_oneandhalf['test_score'])
dt_cv_y_pred_oneandhalf = cross_val_predict(dt_model,x_tv_oneandhalf, y_tv_oneandhalf, cv = cv_model, n_jobs = -1)

dt_cv_accuracy_list_half_one = cross_validate(dt_model,x_tv_half_one, y_tv_half_one, cv = cv_model, n_jobs = -1, return_train_score = True)
dt_cv_train_accuracy_half_one = np.mean(dt_cv_accuracy_list_half_one['train_score'])
dt_cv_test_accuracy_half_one = np.mean(dt_cv_accuracy_list_half_one['test_score'])
dt_cv_y_pred_half_one = cross_val_predict(dt_model,x_tv_half_one, y_tv_half_one, cv = cv_model, n_jobs = -1)

dt_cv_accuracy_list_one_oneandhalf = cross_validate(dt_model,x_tv_one_oneandhalf, y_tv_one_oneandhalf, cv = cv_model, n_jobs = -1, return_train_score = True)
dt_cv_train_accuracy_one_oneandhalf = np.mean(dt_cv_accuracy_list_one_oneandhalf['train_score'])
dt_cv_test_accuracy_one_oneandhalf = np.mean(dt_cv_accuracy_list_one_oneandhalf['test_score'])
dt_cv_y_pred_one_oneandhalf = cross_val_predict(dt_model,x_tv_one_oneandhalf, y_tv_one_oneandhalf, cv = cv_model, n_jobs = -1)

dt_cv_accuracy_list_all = cross_validate(dt_model,x_tv_all, y_tv_all, cv = cv_model, n_jobs = -1, return_train_score = True)
dt_cv_train_accuracy_all = np.mean(dt_cv_accuracy_list_all['train_score'])
dt_cv_test_accuracy_all = np.mean(dt_cv_accuracy_list_all['test_score'])
dt_cv_y_pred_all = cross_val_predict(dt_model,x_tv_all, y_tv_all, cv = cv_model, n_jobs = -1)


rf_cv_accuracy_list_half = cross_validate(rf_model,x_tv_half, y_tv_half, cv = cv_model, n_jobs = -1, return_train_score = True)
rf_cv_train_accuracy_half = np.mean(rf_cv_accuracy_list_half['train_score'])
rf_cv_test_accuracy_half = np.mean(rf_cv_accuracy_list_half['test_score'])
rf_cv_y_pred_half = cross_val_predict(rf_model,x_tv_half, y_tv_half, cv = cv_model, n_jobs = -1)

rf_cv_accuracy_list_one = cross_validate(rf_model,x_tv_one, y_tv_one, cv = cv_model, n_jobs = -1, return_train_score = True)
rf_cv_train_accuracy_one = np.mean(rf_cv_accuracy_list_one['train_score'])
rf_cv_test_accuracy_one = np.mean(rf_cv_accuracy_list_one['test_score'])
rf_cv_y_pred_one = cross_val_predict(rf_model,x_tv_one, y_tv_one, cv = cv_model, n_jobs = -1)

rf_cv_accuracy_list_oneandhalf = cross_validate(rf_model,x_tv_oneandhalf, y_tv_oneandhalf, cv = cv_model, n_jobs = -1, return_train_score = True)
rf_cv_train_accuracy_oneandhalf = np.mean(rf_cv_accuracy_list_oneandhalf['train_score'])
rf_cv_test_accuracy_oneandhalf = np.mean(rf_cv_accuracy_list_oneandhalf['test_score'])
rf_cv_y_pred_oneandhalf = cross_val_predict(rf_model,x_tv_oneandhalf, y_tv_oneandhalf, cv = cv_model, n_jobs = -1)

rf_cv_accuracy_list_half_one = cross_validate(rf_model,x_tv_half_one, y_tv_half_one, cv = cv_model, n_jobs = -1, return_train_score = True)
rf_cv_train_accuracy_half_one = np.mean(rf_cv_accuracy_list_half_one['train_score'])
rf_cv_test_accuracy_half_one = np.mean(rf_cv_accuracy_list_half_one['test_score'])
rf_cv_y_pred_half_one = cross_val_predict(rf_model,x_tv_half_one, y_tv_half_one, cv = cv_model, n_jobs = -1)

rf_cv_accuracy_list_one_oneandhalf = cross_validate(rf_model,x_tv_one_oneandhalf, y_tv_one_oneandhalf, cv = cv_model, n_jobs = -1, return_train_score = True)
rf_cv_train_accuracy_one_oneandhalf = np.mean(rf_cv_accuracy_list_one_oneandhalf['train_score'])
rf_cv_test_accuracy_one_oneandhalf = np.mean(rf_cv_accuracy_list_one_oneandhalf['test_score'])
rf_cv_y_pred_one_oneandhalf = cross_val_predict(rf_model,x_tv_one_oneandhalf, y_tv_one_oneandhalf, cv = cv_model, n_jobs = -1)

rf_cv_accuracy_list_all = cross_validate(rf_model,x_tv_all, y_tv_all, cv = cv_model, n_jobs = -1, return_train_score = True)
rf_cv_train_accuracy_all = np.mean(rf_cv_accuracy_list_all['train_score'])
rf_cv_test_accuracy_all = np.mean(rf_cv_accuracy_list_all['test_score'])
rf_cv_y_pred_all = cross_val_predict(rf_model,x_tv_all, y_tv_all, cv = cv_model, n_jobs = -1)



print("DT:")
print("Half -")
print(dt_train_accuracy_half2, dt_val_accuracy_half2, dt_test_accuracy_half2, dt_cv_train_accuracy_half, dt_cv_test_accuracy_half)
print("\n")

print("One -")
print(dt_train_accuracy_one2, dt_val_accuracy_one2, dt_test_accuracy_one2, dt_cv_train_accuracy_one, dt_cv_test_accuracy_one)
print("\n")

print("Oneandhalf -")
print(dt_train_accuracy_oneandhalf2, dt_val_accuracy_oneandhalf2, dt_test_accuracy_oneandhalf2, dt_cv_train_accuracy_oneandhalf, dt_cv_test_accuracy_oneandhalf)
print("\n")

print("Half_one -")
print(dt_train_accuracy_half_one2, dt_val_accuracy_half_one2, dt_test_accuracy_half_one2, dt_cv_train_accuracy_half_one, dt_cv_test_accuracy_half_one)
print("\n")

print("One_oneandhalf -")
print(dt_train_accuracy_one_oneandhalf2, dt_val_accuracy_one_oneandhalf2, dt_test_accuracy_one_oneandhalf2, dt_cv_train_accuracy_one_oneandhalf, dt_cv_test_accuracy_one_oneandhalf)
print("\n")

print("All -")
print(dt_train_accuracy_all2, dt_val_accuracy_all2, dt_test_accuracy_all2, dt_cv_train_accuracy_all, dt_cv_test_accuracy_all)
print("\n")
print("\n")

print("RF:")
print("Half -")
print(rf_train_accuracy_half2, rf_val_accuracy_half2, rf_test_accuracy_half2, rf_cv_train_accuracy_half, rf_cv_test_accuracy_half)
print("\n")

print("One -")
print(rf_train_accuracy_one2, rf_val_accuracy_one2, rf_test_accuracy_one2, rf_cv_train_accuracy_one, rf_cv_test_accuracy_one)
print("\n")

print("Oneandhalf -")
print(rf_train_accuracy_oneandhalf2, rf_val_accuracy_oneandhalf2, rf_test_accuracy_oneandhalf2, rf_cv_train_accuracy_oneandhalf, rf_cv_test_accuracy_oneandhalf)
print("\n")

print("Half_one -")
print(rf_train_accuracy_half_one2, rf_val_accuracy_half_one2, rf_test_accuracy_half_one2, rf_cv_train_accuracy_half_one, rf_cv_test_accuracy_half_one)
print("\n")

print("One_oneandhalf -")
print(rf_train_accuracy_one_oneandhalf2, rf_val_accuracy_one_oneandhalf2, rf_test_accuracy_one_oneandhalf2, rf_cv_train_accuracy_one_oneandhalf, rf_cv_test_accuracy_one_oneandhalf)
print("\n")

print("All -")
print(rf_train_accuracy_all2, rf_val_accuracy_all2, rf_test_accuracy_all2, rf_cv_train_accuracy_all, rf_cv_test_accuracy_all)

fig01, axes01 = plt.subplots(2, 3) 
dt_cv_cm_half = confusion_matrix(y_tv_half,dt_cv_y_pred_half)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cv_cm_half,display_labels=dt_model.classes_)
disp.plot(ax = axes01[0,0])
disp.ax_.set_title("Distance = 0.5m")

dt_cv_cm_one = confusion_matrix(y_tv_one,dt_cv_y_pred_one)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cv_cm_one,display_labels=dt_model.classes_)
disp.plot(ax = axes01[0,1])
disp.ax_.set_title("Distance = 1m")

dt_cv_cm_oneandhalf = confusion_matrix(y_tv_oneandhalf,dt_cv_y_pred_oneandhalf)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cv_cm_oneandhalf,display_labels=dt_model.classes_)
disp.plot(ax = axes01[0,2])
disp.ax_.set_title("Distance = 1.5m")

dt_cv_cm_half_one = confusion_matrix(y_tv_half_one,dt_cv_y_pred_half_one)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cv_cm_half_one,display_labels=dt_model.classes_)
disp.plot(ax = axes01[1,0])
disp.ax_.set_title("Distance = 0.5m and 1m")

dt_cv_cm_one_oneandhalf = confusion_matrix(y_tv_one_oneandhalf,dt_cv_y_pred_one_oneandhalf)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cv_cm_one_oneandhalf,display_labels=dt_model.classes_)
disp.plot(ax = axes01[1,1])
disp.ax_.set_title("Distance = 1m and 1.5m")

dt_cv_cm_all = confusion_matrix(y_tv_all,dt_cv_y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cv_cm_all,display_labels=dt_model.classes_)
disp.plot(ax = axes01[1,2])
disp.ax_.set_title("Distance = 0.5m, 1m and 1.5m")

fig0, axes0 = plt.subplots(2, 3) 
rf_cv_cm_half = confusion_matrix(y_tv_half,rf_cv_y_pred_half)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cv_cm_half,display_labels=rf_model.classes_)
disp.plot(ax = axes0[0,0])
disp.ax_.set_title("Distance = 0.5m")

rf_cv_cm_one = confusion_matrix(y_tv_one,rf_cv_y_pred_one)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cv_cm_one,display_labels=rf_model.classes_)
disp.plot(ax = axes0[0,1])
disp.ax_.set_title("Distance = 1m")

rf_cv_cm_oneandhalf = confusion_matrix(y_tv_oneandhalf,rf_cv_y_pred_oneandhalf)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cv_cm_oneandhalf,display_labels=rf_model.classes_)
disp.plot(ax = axes0[0,2])
disp.ax_.set_title("Distance = 1.5m")

rf_cv_cm_half_one = confusion_matrix(y_tv_half_one,rf_cv_y_pred_half_one)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cv_cm_half_one,display_labels=rf_model.classes_)
disp.plot(ax = axes0[1,0])
disp.ax_.set_title("Distance = 0.5m and 1m")

rf_cv_cm_one_oneandhalf = confusion_matrix(y_tv_one_oneandhalf,rf_cv_y_pred_one_oneandhalf)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cv_cm_one_oneandhalf,display_labels=rf_model.classes_)
disp.plot(ax = axes0[1,1])
disp.ax_.set_title("Distance = 1m and 1.5m")

rf_cv_cm_all = confusion_matrix(y_tv_all,rf_cv_y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cv_cm_all,display_labels=rf_model.classes_)
disp.plot(ax = axes0[1,2])
disp.ax_.set_title("Distance = 0.5m, 1m and 1.5m")

fig1, axes1 = plt.subplots(2, 3) 
#plt.figure(figsize=(12,10))
k = range(param1_start,param1_end+1)
axes1[0,0].plot(k, dt_train_accuracy_half*100,k,dt_val_accuracy_half*100)
axes1[0,0].grid()
axes1[0,0].set_xlabel("Maximum depth of decision tree")
axes1[0,0].set_ylabel("Accuracy (%)")
axes1[0,0].set_title("Distance = 0.5m")
axes1[0,0].legend(["Train accuracy", "Validation accuracy"])

axes1[0,1].plot(k,dt_train_accuracy_one*100,k,dt_val_accuracy_one*100)
axes1[0,1].grid()
axes1[0,1].set_xlabel("Maximum depth of decision tree")
axes1[0,1].set_ylabel("Accuracy (%)")
axes1[0,1].set_title("Distance = 1m")
axes1[0,1].legend(["Train accuracy", "Validation accuracy"])

axes1[0,2].plot(k, dt_train_accuracy_oneandhalf*100,k,dt_val_accuracy_oneandhalf*100)
axes1[0,2].set_xlabel("Maximum depth of decision tree")
axes1[0,2].set_ylabel("Accuracy (%)")
axes1[0,2].set_title("Distance = 1.5m")
axes1[0,2].legend(["Train accuracy", "Validation accuracy"])
axes1[0,2].grid()

axes1[1,0].plot(k, dt_train_accuracy_half_one*100,k,dt_val_accuracy_half_one*100)
axes1[1,0].set_xlabel("Maximum depth of decision tree")
axes1[1,0].set_ylabel("Accuracy (%)")
axes1[1,0].set_title("Distance = 0.5m and 1m")
axes1[1,0].legend(["Train accuracy", "Validation accuracy"])
axes1[1,0].grid()

axes1[1,1].plot(k, dt_train_accuracy_one_oneandhalf*100,k,dt_val_accuracy_one_oneandhalf*100)
axes1[1,1].set_xlabel("Maximum depth of decision tree")
axes1[1,1].set_ylabel("Accuracy (%)")
axes1[1,1].set_title("Distance = 1m and 1.5m")
axes1[1,1].legend(["Train accuracy", "Validation accuracy"])
axes1[1,1].grid()

axes1[1,2].plot(k, dt_train_accuracy_all*100,k,dt_val_accuracy_all*100)
axes1[1,2].set_xlabel("Maximum depth of decision tree")
axes1[1,2].set_ylabel("Accuracy (%)")
axes1[1,2].set_title("Distance = 0.5m, 1m and 1.5m")
axes1[1,2].legend(["Train accuracy", "Validation accuracy"])
axes1[1,2].grid()


fig2, axes2 = plt.subplots(2, 3) 
#plt.figure(figsize=(12,10))
k = range(param1_start,param1_end+1)
axes2[0,0].plot(k, rf_train_accuracy_half*100,k,rf_val_accuracy_half*100)
axes2[0,0].grid()
axes2[0,0].set_xlabel("Number of decision trees")
axes2[0,0].set_ylabel("Accuracy (%)")
axes2[0,0].set_title("Distance = 0.5m")
axes2[0,0].legend(["Train accuracy", "Validation accuracy"])

axes2[0,1].plot(k,rf_train_accuracy_one*100,k,rf_val_accuracy_one*100)
axes2[0,1].grid()
axes2[0,1].set_xlabel("Number of decision trees")
axes2[0,1].set_ylabel("Accuracy (%)")
axes2[0,1].set_title("Distance = 1m")
axes2[0,1].legend(["Train accuracy", "Validation accuracy"])

axes2[0,2].plot(k, rf_train_accuracy_oneandhalf*100,k,rf_val_accuracy_oneandhalf*100)
axes2[0,2].set_xlabel("Number of decision trees")
axes2[0,2].set_ylabel("Accuracy (%)")
axes2[0,2].set_title("Distance = 1.5m")
axes2[0,2].legend(["Train accuracy", "Validation accuracy"])
axes2[0,2].grid()

axes2[1,0].plot(k, rf_train_accuracy_half_one*100,k,rf_val_accuracy_half_one*100)
axes2[1,0].set_xlabel("Number of decision trees")
axes2[1,0].set_ylabel("Accuracy (%)")
axes2[1,0].set_title("Distance = 0.5m and 1m")
axes2[1,0].legend(["Train accuracy", "Validation accuracy"])
axes2[1,0].grid()

axes2[1,1].plot(k, rf_train_accuracy_one_oneandhalf*100,k,rf_val_accuracy_one_oneandhalf*100)
axes2[1,1].set_xlabel("Number of decision trees")
axes2[1,1].set_ylabel("Accuracy (%)")
axes2[1,1].set_title("Distance = 1m and 1.5m")
axes2[1,1].legend(["Train accuracy", "Validation accuracy"])
axes2[1,1].grid()

axes2[1,2].plot(k, rf_train_accuracy_all*100,k,rf_val_accuracy_all*100)
axes2[1,2].set_xlabel("Number of decision trees")
axes2[1,2].set_ylabel("Accuracy (%)")
axes2[1,2].set_title("Distance = 0.5m, 1m and 1.5m")
axes2[1,2].legend(["Train accuracy", "Validation accuracy"])
axes2[1,2].grid()

plt.show()