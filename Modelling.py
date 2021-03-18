#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, auc, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('train_bert_PCA.csv')
test = pd.read_csv('test_bert_PCA.csv')

yTrain = train['engagement_rate_label'].to_numpy()
xTrain = train.drop(['engagement_rate_label'], axis=1)

yTest = test['engagement_rate_label'].to_numpy()
xTest = test.drop(['engagement_rate_label'], axis=1)


def model(mode):
    if mode == 'dummy':
        clf = DummyClassifier(strategy="stratified") # used for baseline comparison
    elif mode == 'logistic':
        clf = LogisticRegression(random_state=0, max_iter=5000)
    elif mode == 'rf':
        clf = RandomForestClassifier(n_estimators=220, min_samples_leaf=45, max_features='log2',
                                     max_depth=125, criterion='entropy')
    elif mode == 'knn':
        clf = KNeighborsClassifier(weights='uniform', n_neighbors=85,
                                   metric='euclidean')
    elif mode == 'nb':
        clf = GaussianNB()
    clf.fit(xTrain, yTrain)
    predictions = clf.predict(xTest)
    print(classification_report(predictions, yTest))

def param_tuning(mode):
    if mode == 'rf':
        params = {'max_depth': [x for x in range(1, 1000) if x % 5 == 0], 
                  'max_features': ['auto', 'log2'],
                  'min_samples_leaf': [x for x in range(1,50) if x % 3 == 0],
                  'n_estimators': [x for x in range(10, 300) if x % 10 == 0],
                  'criterion': ['gini', 'entropy']}
        model = RandomForestClassifier()
    elif mode == 'knn':
        params = {'n_neighbors': [x for x in range(1,200) if x % 5 == 0],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'minkowski']}
        model = KNeighborsClassifier()
    clf = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='f1_weighted')
    clf.fit(xTrain, yTrain)
    print("best params: ", clf.best_params_)
    print('best_score', clf.best_score_)
    
if __name__ == '__main__':
    model('logistic')
    # param_tuning('knn')