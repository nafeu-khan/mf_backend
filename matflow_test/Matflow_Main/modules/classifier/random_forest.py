import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    if do_hyperparameter_optimization:
        st.subheader("Hyperparameter Optimization Settings")
        n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=5, step=1)
        cv = st.number_input("Number of cross-validation folds", min_value=2, value=2, step=1)
        random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    if "rf_best_param" not in st.session_state:
        st.session_state.rf_best_param = {
            "criterion":"gini",
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "auto",
            "random_state": 0
        }

    st.write('#')

    if do_hyperparameter_optimization and st.button('Run Optimization'):

        st.write('#')
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

        with st.spinner('Doing hyperparameter optimization...'):

            clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                                     random_state=random_state)
            st.spinner("Fitting the model...")
            clf.fit(X_train, y_train)

        st.success("Hyperparameter optimization completed!")

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

        if "opti_results_df" not in st.session_state:
            st.session_state.opti_results_df = pd.DataFrame

        st.session_state.opti_results_df = results_df

        best_param = clf.best_params_

        st.session_state.rf_best_param = best_param

    return st.session_state.rf_best_param


def random_forest(X_train, y_train):

    best_param = hyperparameter_optimization(X_train, y_train)

    try:
        st.write('#')
        st.write(st.session_state.opti_results_df)
    except:
        pass


    st.subheader("Model Settings")

    max_depth = best_param['max_depth']
    col1, col2, col3 = st.columns(3)
    n_estimators = col1.number_input(
        "Number of Estimators",
        1, 10000, best_param['n_estimators'],
        key="rf_n_estimators"
    )

    criterion = col2.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=["gini", "entropy", "log_loss"].index(best_param['criterion']),
        key="rf_criterion"
    )

    min_samples_split = col3.number_input(
        "Min. Samples Split",
        2, 20, best_param['min_samples_split'],
        key="rf_min_samples_split"
    )

    col1, col2, col3, col4 = st.columns([3.33, 2, 1.33, 3.33])
    min_samples_leaf = col1.number_input(
        "Min. Samples Leaf",
        1, 20, best_param['min_samples_leaf'],
        key="rf_min_samples_leaf"
    )

    col3.markdown("#")
    auto_max_depth = col3.checkbox("Auto", max_depth is None, key="rf_auto_max_depth")

    if auto_max_depth:
        max_depth = None
        max_depth_input = col2.text_input(
            "Max Depth",
            "None",
            key="rf_max_depth_none",
            disabled=True
        )
    else:
        max_depth_input = col2.number_input(
            "Max Depth",
            1, 20, max_depth,best_param['max_depth'],
            key="rf_max_depth"
        )

    random_state = col4.number_input(
        "Random State",
        0, 1000000, best_param['random_state'],
        key="rf_random_state"
    )

    max_depth = None if auto_max_depth else max_depth
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=random_state)
    except Exception as e:
        print("An error occurred while initializing the RandomForestClassifier model:", e)
    # Handle the error in an appropriate way (e.g., log the error, display a message to the user, etc.)

    return model
