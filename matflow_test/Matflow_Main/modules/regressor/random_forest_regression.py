import time

import pandas as pd
import streamlit as st
from django.http import JsonResponse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def hyperparameter_optimization(X_train, y_train,file):
    n_iter = int(file.get("Number of iterations for hyperparameter search"))
    cv = int(file.get("Number of cross-validation folds"))
    random_state = int(file.get("Random state for hyperparameter search"))

    param_dist = {
        "n_estimators": [10, 50, 100],
        "criterion": ["friedman_mse", "squared_error", "absolute_error", "poisson"],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [3, 5, 10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "n_jobs": [-1],
    }
    model = RandomForestRegressor(random_state=0)
    clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                             random_state=random_state)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    rows = []
    for param in best_params:
        rows.append([param, best_params[param]])
    table = pd.DataFrame(rows, columns=['Parameter', 'Value'])
    table=table.to_dict(orient="records")
    obj = {
        "result": table,  # table
        "param": best_params  # parameter
    }
    return JsonResponse(obj)


def random_forest_regressor(X_train, y_train,file):

    n_estimators = int(file.get("Number of trees"))
    max_features = file.get("Max features")
    min_samples_split = int(file.get("Min. samples split"))
    max_depth = int(file.get("Max depth"))
    random_state = int(file.get("Random state"))
    min_samples_leaf = int(file.get("Min. samples leaf"))
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=random_state)

    return model
