import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    if do_hyperparameter_optimization:
        st.subheader("Hyperparameter Optimization Settings")
        n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=5, step=1)
        cv = st.number_input("Number of cross-validation folds", min_value=2, value=2, step=1)
        random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    if "logreg_best_param" not in st.session_state:
        st.session_state.logreg_best_param = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 1e-4,
            "random_state": 42
        }

    st.write('#')

    if do_hyperparameter_optimization and st.button('Run Optimization'):

        st.write('#')
        param_dist = {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            "max_iter": [100, 200, 300, 400, 500],
            "tol": [1e-4, 1e-3, 1e-2],
            "random_state": [42]
        }
        model = LogisticRegression()

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

        st.session_state.logreg_best_param = best_param

    return st.session_state.logreg_best_param


def log_reg(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    try:
        st.write('#')
        st.write(st.session_state.opti_results_df)
    except:
        pass

    st.subheader("Model Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        penalty = st.selectbox(
            "Penalty",
            ["none", "l1", "l2", "elasticnet"],
            index=["none", "l1", "l2", "elasticnet"].index(best_params['penalty']),
            # set the initial value from best_param
            key="lr_penalty"
        )

        solver = st.selectbox(
            "Solver",
            ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            index=["newton-cg", "lbfgs", "liblinear", "sag", "saga"].index(best_params['solver']),
            # set the initial value from best_param
            key="lr_solver"
        )

    with col2:
        C = st.number_input(
            "C",
            0.01, 1000.0, float(best_params['C']), 0.01,  # set the initial value from best_param
            format="%f",
            key="lr_c"
        )

        max_iter = st.number_input(
            "Max Iterations",
            1, 1000000, best_params['max_iter'],  # set the initial value from best_param
            key="lr_max_iter"
        )

    with col3:
        tol = st.number_input(
            "Tolerance (ε)",
            1e-8, 1.0, best_params['tol'],  # set the initial value from best_param
            format="%f",
            key="lr_tol"
        )

        random_state = st.number_input(
            "Random State",
            0, 1000000, best_params['random_state'],  # set the initial value from best_param
            key="lr_random_state"
        )

    try:
        model = LogisticRegression(penalty=penalty, C=C, tol=tol, solver=solver, max_iter=max_iter,
                                   random_state=random_state)
    except ValueError as e:
        print(f"Error: {e}")
    # Handle the ValueError exception here
    except TypeError as e:
        print(f"Error: {e}")
    # Handle the TypeError exception here
    except Exception as e:
        print(f"Error: {e}")
    # Handle any other exception here

    return model
