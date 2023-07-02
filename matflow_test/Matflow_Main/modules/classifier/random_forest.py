import time

import pandas as pd
import streamlit as st
from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train,file):
    # do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    # if do_hyperparameter_optimization:
    n_iter =file.get("Number of iterations for hyperparameter search")
    cv = file.get("Number of cross-validation folds")
    random_state = file.get("Random state for hyperparameter search")
    #
    # if "rf_best_param" not in st.session_state:
    #     st.session_state.rf_best_param = {
    #         "criterion":"gini",
    #         "n_estimators": 100,
    #         "max_depth": None,
    #         "min_samples_split": 2,
    #         "min_samples_leaf": 1,
    #         "max_features": "auto",
    #         "random_state": 0
    #     }

    param_dist = {
        "criterion": ["gini", "entropy", "log_loss"],
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "random_state": [0]
    }
    model = RandomForestClassifier()
    # with st.spinner('Doing hyperparameter optimization...'):
    clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                             random_state=random_state)
    # st.spinner("Fitting the model...")
    clf.fit(X_train, y_train)
    cv_results = clf.cv_results_
    param_names = list(cv_results['params'][0].keys())

    # Create a list of dictionaries with the parameter values and accuracy score for each iteration
    results_list = []
    for i in range(len(cv_results['params'])):
        param_dict = {}
        for param in param_names:
            param_dict[param] = cv_results['params'][i][param]
        param_dict['accuracy'] = cv_results['mean_test_score'][i]
        results_list.append(param_dict)

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by=['accuracy'], ascending=False)
    best_param = clf.best_params_
    return JsonResponse(best_param)


def random_forest(X_train, y_train,file):
    best_param = hyperparameter_optimization(X_train, y_train,file)
#("Model Settings")

    max_depth = best_param['max_depth']
    n_estimators = file.get("Number of Estimators")

    criterion =file.get("Criterion")

    min_samples_split = file.get("Min. Samples Split")

    min_samples_leaf =file.get("Min. Samples Leaf")

    auto_max_depth = file.get("Auto")

    if auto_max_depth:
        max_depth = None
        max_depth_input =file.get( "Max Depth")
    else:
        max_depth_input = file.get("Max Depth")
    random_state = file.get("Random State")

    max_depth = None if auto_max_depth else max_depth
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=random_state)
    except Exception as e:
        print("An error occurred while initializing the RandomForestClassifier model:", e)
    # Handle the error in an appropriate way (e.g., log the error, display a message to the user, etc.)

    return model
