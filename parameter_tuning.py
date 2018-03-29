# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:48:11 2018

@author: hp
"""


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
digits = load_digits()

def test_train_split(test_size_value):
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size_value, random_state=0)
    return x_train,x_test,y_train,y_test

split_size=[0.2,0.25,0.3,0.35,0.4]
'''
function for evaluation and accuracy measures
'''

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy= accuracy_score(test_labels, predictions)
    cm=confusion_matrix(test_labels, predictions)
    print('Confusion Matrix')
    print(cm)
    print('\n')
    print('Classification Report')
    print('\n')
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']
    print(classification_report(test_labels, predictions, target_names=target_names))
    print('Accuracy')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('\n')
    
    print('area under ROC curve')
    lb = preprocessing.LabelBinarizer()
    y_test_b=lb.fit_transform(test_labels)
    probs = clf.predict_proba(test_features)
    preds = probs[:,1]
    y_test_bs = y_test_b[:,1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_bs, preds)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    print(roc_auc)
    print('\n')
    print('RoC curve')
    

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

"""
Implementation of Decision tree using Grid Search

"""
for i in split_size:
   x_train,x_test,y_train,y_test=test_train_split(i)
   dt = DecisionTreeClassifier()
   parameters = {"max_depth": [None, 2, 5, 10], "min_samples_leaf": [1, 5, 10],"max_leaf_nodes": [None, 5, 10, 20],"min_samples_split": [2, 10, 20]}   
   clf = GridSearchCV(dt, parameters)
   clf.fit(x_train, y_train)
   evaluate(clf, x_test, y_test)




print(clf.best_params_)

"""
Implementation of Neural nets using Grid Search
"""

for i in split_size:
   x_train,x_test,y_train,y_test=test_train_split(i)
   mlp = MLPClassifier()
   parameters = {"activation": ['logistic','relu'], "alpha": [0.1,0.2],"learning_rate": ['constant','adaptive'],"max_iter": [300,500]}   
   clf = GridSearchCV(mlp, parameters)
   clf.fit(x_train, y_train)
   evaluate(clf, x_test, y_test)


print(clf.best_params_)
   
"""
Implementaton of K-nearest neighbors using Grid Search
"""

for i in split_size:
   x_train,x_test,y_train,y_test=test_train_split(i)
   knn = KNeighborsClassifier()
   parameters = {"n_neighbors": [5,6,7], "algorithm": ['ball_tree','auto','brute'],"p": [2,3],"weights": ['uniform','distance']}   
   clf = GridSearchCV(knn, parameters)
   clf.fit(x_train, y_train)
   evaluate(clf, x_test, y_test)   



print(clf.best_params_)

"""
Implementation of SVM using Grid Search
"""

for i in split_size:
   x_train,x_test,y_train,y_test=test_train_split(i)
   svc = SVC(probability = True)
   parameters = {"C": [1], "kernel": ['linear','poly', 'rbf'],"max_iter": [-1]}
   clf = GridSearchCV(svc, parameters)
   clf.fit(x_train, y_train)
   evaluate(clf, x_test, y_test)


print(clf.best_params_)


"""
Implementation of Gaussian NB using Grid Search
"""
for i in split_size:
   x_train,x_test,y_train,y_test=test_train_split(i)
   Gb = GaussianNB(priors=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
   parameters = {}  
   clf = GridSearchCV(Gb, parameters) 
   clf.fit(x_train, y_train)
   evaluate(clf, x_test, y_test)



print(clf.best_params_)

"""
Implementation of Logistic Regression using Grid Search
"""

for i in split_size:
   x_train,x_test,y_train,y_test=test_train_split(i)
   logistic = LogisticRegression()
   parameters = {"penalty": ["l1","l2"], "C": [1.0,0.5],"fit_intercept": ['True','False'],"max_iter": [300,500]}   
   clf = GridSearchCV(logistic, parameters)
   clf.fit(x_train, y_train)
   evaluate(clf, x_test, y_test)


print(clf.best_params_)
