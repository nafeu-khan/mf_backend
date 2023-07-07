import json

import pandas as pd
import streamlit as st
import pickle

from django.http import JsonResponse

from ...modules.utils import split_xy
from ...modules.regressor import linear_regression, ridge_regression, lasso_regression, decision_tree_regression, random_forest_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def regression(file):
    dataset=pd.DataFrame(file.get("file"))
    train_data = pd.DataFrame(file.get("train"))
    test_data = pd.DataFrame(file.get("test"))
    target_var = file.get("target_var")
    X_train, y_train = split_xy(train_data, target_var)
    X_test, y_test = split_xy(test_data, target_var)

    try:
        X_train, X_test = X_train.drop(target_var, axis=1), X_test.drop(target_var, axis=1)
    except:
        pass

    regressors = {
        "Linear Regression": "LR",
        "Ridge Regression": "Ridge",
        "Lasso Regression": "Lasso",
        "Decision Tree Regression": "DT",
        "Random Forest Regression": "RF"
    }
    metrics= ["R-Squared", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error"]

    regressor = file.get("Regressor")
    # model_name = col2.text_input(
    #         "Model Name",
    #         f"{regressors[regressor]}_Regression",
    #         key="model_name"
    #     )

    if regressor == "Linear Regression":
        model = linear_regression.linear_regression(X_train, y_train)
    elif regressor == "Ridge Regression":
        model = ridge_regression.ridge_regression(X_train, y_train)
    elif regressor == "Lasso Regression":
        model = lasso_regression.lasso_regression(X_train, y_train)
    elif regressor == "Decision Tree Regression":
        model = decision_tree_regression.decision_tree_regressor(X_train, y_train)
    elif regressor == "Random Forest Regression":
        model = random_forest_regression.random_forest_regressor(X_train, y_train)


    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return JsonResponse({'error': "model doesn't fitted"})

    X, y = split_xy(dataset, target_var)
    y_prediction = model.predict(X)

    selected_metrics = get_result(model, X_test, y_test, metrics)
    for met in metrics:
        st.metric(met, selected_metrics.get(met))
    i = 0
    for X, y in zip([X_train, X_test], [y_train, y_test]):
        list2 = get_result(model, X, y, metrics)
        if (i == 0):
            list1 = get_result(model, X, y, metrics)
        i += 1

    merged_list = {
        f"Train {key}": value
        for key, value in list1.items()
    }

    merged_list.update({
        f"Test {key}": value
        for key, value in list2.items()
    })
    y_prediction=json.dumps(y_prediction.tolist())
    obj={
        "metrics": selected_metrics,   #4
        "metrics_table":merged_list,     #8
        "y_pred" : y_prediction
    }
    return JsonResponse(obj)

def get_result(model, X, y, metrics):
    y_pred = model.predict(X)
    metric_dict = {
        "R-Squared": r2_score(y, y_pred),
        "Mean Absolute Error": mean_absolute_error(y, y_pred),
        "Mean Squared Error": mean_squared_error(y, y_pred),
        "Root Mean Squared Error": mean_squared_error(y, y_pred, squared=False)
    }
    result=metric_dict

    return result
