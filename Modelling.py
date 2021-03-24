#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, auc, accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

mode = 'shares_label3'

train = pd.read_csv('train_new2.csv')
test = pd.read_csv('test_new2.csv')

yTrain = train[mode].to_numpy()
xTrain = train.drop(columns=[mode], axis=1)

yTest = test[mode].to_numpy()
xTest = test.drop(columns=[mode], axis=1)

predictors = list(xTrain.columns)


def model(mode):
    if mode == 'dummy':
        clf = DummyClassifier(strategy="uniform") # used for baseline comparison
    elif mode == 'logistic':
        clf = LogisticRegression(random_state=0, max_iter=3000)
    elif mode == 'rf':
        clf = RandomForestClassifier(n_estimators=80, min_samples_leaf=6, max_features='auto',
                                     max_deth=295, criterion='entropy')
    elif mode == 'knn':
        clf = KNeighborsClassifier(weights='uniform', n_neighbors=15,
                                   metric='euclidean')
    elif mode == 'svm':
        clf = SVC(C=10, gamma=0.01, kernel='rbf')
    elif mode == 'gb':
        clf = GradientBoostingClassifier(n_estimators=128, min_samples_split=0.5, min_samples_leaf=0.2,
                                         max_features=24, learning_rate=0.25, max_depth=9)
    clf.fit(xTrain, yTrain)
    predictions = clf.predict(xTest)
    print(classification_report(predictions, yTest))
    print('Accuracy: ', accuracy_score(yTest, predictions))
    
    fpr, tpr, thresholds = roc_curve(yTest, predictions)
    print('AUC: ', auc(fpr, tpr))

    #### available for Random Forest, Gradient Boosting (Feature Importances) and Logistic Regression (Feature Coefficients)
    feat_imp = pd.Series(clf.coef_[0], predictors).sort_values(ascending=False) # coef_[0] or feature_importances_
    feat_imp.plot(kind='bar', title='Feature Coefficients (Logistic Regression)' ) # Feature Coefficients or Feature Importances
    plt.ylabel('Coefficient of the features in the decision function') # Coefficient of the features in the decision function or Feature Importance Score
    plt.show()

def param_tuning(mode, search, metric):
    if mode == 'rf':
        params = {'max_depth': [x for x in range(1, 1000) if x % 5 == 0], 
                  'max_features': ['auto', 'log2'],
                  'min_samples_leaf': [x for x in range(1,50) if x % 3 == 0],
                  'n_estimators': [x for x in range(10, 300) if x % 10 == 0],
                  'criterion': ['gini', 'entropy']}
        model = RandomForestClassifier()
    elif mode == 'knn':
        params = {'n_neighbors': [x for x in range(1,50) if x % 5 == 0],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'minkowski']}
        model = KNeighborsClassifier()
    elif mode == 'svm':
        params = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001]}
        model = SVC()
    elif mode == 'gb':
        params = {'learning_rate': [1, 0.5, 0.25, 0.1, 0.01], 
                  'max_depth': [1, 3, 6, 9, 12, 15, 18, 21, 24, 27],
                  'min_samples_split': [0.1, 0.3, 0.5, 0.7],
                  'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
                  'max_features': [1, 4, 8, 12, 16, 20, 24, 28],
                  'n_estimators': [16, 32, 64, 128, 256]}
        model = GradientBoostingClassifier()
    if search == 'grid':
        clf = GridSearchCV(estimator=model, param_grid=params, scoring=metric)
    elif search == 'random':
        clf = RandomizedSearchCV(estimator=model, param_distributions=params, scoring=metric) 
    clf.fit(xTrain, yTrain)
    print("best params: ", clf.best_params_)
    print('best_score (%s):' %(metric), clf.best_score_)
    
if __name__ == '__main__':
    # model('')
    param_tuning('rf', 'random', 'f1_weighted')