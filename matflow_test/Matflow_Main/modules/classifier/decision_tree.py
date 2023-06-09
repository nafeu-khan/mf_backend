import time

import pandas as pd
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


def hyperparameter_optimization(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    if do_hyperparameter_optimization:
        st.subheader("Hyperparameter Optimization Settings")
        n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=5, step=1)
        cv = st.number_input("Number of cross-validation folds", min_value=2, value=2, step=1)
        random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)


    if "dt_best_param" not in st.session_state:
        st.session_state.dt_best_param = {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "random_state": random_state if not random_state==None else 0
        }

    st.write('#')

    if do_hyperparameter_optimization and st.button('Run Optimization'):

        st.write('#')
        param_dist = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 1, 2, 3, 4, 5, 10, 20, 50, 100],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [2, 4, 8, 10],
            "random_state": [random_state]

        }
        model = DecisionTreeClassifier()

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

        st.session_state.dt_best_param = best_param

    return st.session_state.dt_best_param


def decision_tree(X_train, y_train):
    best_param = hyperparameter_optimization(X_train, y_train)
    try:
        st.write(st.session_state.opti_results_df)
    except:
        pass

    st.subheader("Model Settings")

    max_depth = None
    col1, col2, col3 = st.columns(3)
    criterion = col1.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        index=["gini", "entropy", "log_loss"].index(best_param['criterion']) if best_param else 0,
        key="dt_criterion"
    )

    min_samples_split = col2.number_input(
        "Min. Samples Split",
        min_value=2, max_value=20, step=1,
        value=best_param['min_samples_split'] if best_param else 2,
        key="dt_min_samples_split"
    )
    min_samples_leaf = col3.number_input(
        "Min. Samples Leaf",
        min_value=2, max_value=20, step=1,
        value=best_param['min_samples_leaf'] if best_param else 1,
        key="dt_min_samples_leaf"
    )

    col1.markdown("#")

    auto_max_depth = col1.checkbox("None", True, key="dt_auto_max_depth")
    if auto_max_depth:
        max_depth = None
    else:
        max_depth = col2.number_input(
            "Max Depth",
            1, 20, 7,
            value=best_param['max_depth'] if best_param else 7,
            key="dt_max_depth"
        )

    random_state = col3.number_input(
        "Random State",
        min_value=0, max_value=100, step=1,
        value=best_param['random_state'] if best_param else 0,
        key="dt_random_state"
    )

    max_depth = None if auto_max_depth else max_depth

    try:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=random_state)

    except ValueError as e:
        print("ValueError occurred while initializing the model:", e)
    except TypeError as e:
        print("TypeError occurred while initializing the model:", e)
    except:
        print("An error occurred while initializing the model.")

    return model
