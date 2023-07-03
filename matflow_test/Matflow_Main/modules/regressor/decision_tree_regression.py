import time

import pandas as pd
import streamlit as st
from django.http import JsonResponse
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
def hyperparameter_optimization(X_train, y_train,file):
    n_iter = int(file.get("Number of iterations for hyperparameter search"))
    cv = int(file.get("Number of cross-validation folds"))
    random_state = int(file.get("Random state for hyperparameter search"))


    param_dist = {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["mse", "friedman_mse", "mae"],
        "random_state": [random_state],
    }
    model = DecisionTreeRegressor()

    for i in range(100):
        time.sleep(0.1)
        st.spinner(f"Running iteration {i + 1} of {n_iter}...")
    clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                             random_state=random_state)
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    rows = []
    for param in best_params:
        rows.append([param, best_params[param]])
    # st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))

    # return clf.best_estimator_

    best_param = clf.best_params_
    return JsonResponse(best_param)


def decision_tree_regressor(X_train, y_train,file):


    max_depth = None
    col1, col2, col3 = st.columns(3)
    criterion = file.get("Criterion")

    min_samples_split =file.get("Min. Samples Split")

    min_samples_leaf = file.get( "Min. Samples Leaf")

    auto_max_depth = file.get("auto_max_depth")
    if auto_max_depth:
        max_depth = col1.text_input(
            "Max Depth",
            None,
            key="dtr_max_depth_none",
            disabled=True
        )
    else:
        max_depth = file.get("Max Depth")

    random_state = file.get("Random State")

    max_depth = None if auto_max_depth else max_depth
    from sklearn.tree import DecisionTreeRegressor

    try:
        model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf, random_state=random_state)
    except ValueError as e:
        print(f"Error creating DecisionTreeRegressor: {str(e)}")
    except:
        print("Unexpected error creating DecisionTreeRegressor")

    return model
