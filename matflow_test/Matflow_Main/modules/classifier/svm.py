import pandas as pd
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC


def hyperparameter_optimization(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    if do_hyperparameter_optimization:
        st.subheader("Hyperparameter Optimization Settings")
        n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=5, step=1)
        cv = st.number_input("Number of cross-validation folds", min_value=2, value=2, step=1)
        random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    if "svm_best_param" not in st.session_state:
        st.session_state.svm_best_param = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "degree": 3
        }

    st.write("#")

    if do_hyperparameter_optimization and st.button('Run Optimization'):

        st.write("#")
        param_dist = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4, 5]
        }
        model = SVC()

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

        st.session_state.svm_best_param = best_param

    return st.session_state.svm_best_param


def svm(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    try:
        st.write("#")
        st.write(st.session_state.opti_results_df)
    except:
        pass

    st.subheader("Model Settings")

    col1, col2, col3 = st.columns(3)
    C = col1.number_input(
        "C",
        0.01, 1000.0, float(best_params['C']), 0.01,
        key="svm_c"
    )

    kernel = col2.selectbox(
        "Kernel",
        ["linear", "poly", "rbf", "sigmoid"],
        ["linear", "poly", "rbf", "sigmoid"].index(best_params.get('kernel', 'rbf')),
        key="svm_kernel"
    )

    tol = col3.number_input(
        "Tolerance (ε)",
        0.000001, 170.0, float(best_params.get('tol', 0.001)), 0.001,
        format="%f",
        key="svm_tol"
    )

    col1, col2, col3 = st.columns(3)

    gamma = col1.selectbox(
        "Gamma",
        ["scale", "auto"],
        ["scale", "auto"].index(best_params.get('gamma', 'scale')),
        key="svm_gamma"
    )


    degree = col3.number_input(
        "Polinomial Degree",
        1, 100, best_params.get('degree', 3), 1,
        key="svm_degree"
    )

    try:
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, tol=tol)
    except ValueError as ve:
        print("ValueError:", ve)
    except TypeError as te:
        print("TypeError:", te)
    except:
        print("An unexpected error occurred.")

    return model
