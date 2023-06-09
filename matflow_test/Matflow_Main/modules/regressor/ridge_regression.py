import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
import time
import pandas as pd

def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')

    if st.button('Run Optimization'):

        st.write('#')
        param_dist = {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
            "fit_intercept": [True, False],
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            "max_iter": [100, 500, 1000],
            "tol": [0.0001, 0.001, 0.01, 0.1, 1],
            "random_state": [0]
        }
        model = Ridge()

        with st.spinner('Doing hyperparameter optimization...'):
            for i in range(100):
                time.sleep(0.1)
                st.spinner(f"Running iteration {i + 1} of {n_iter}...")
            clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                                     random_state=random_state)
            st.spinner("Fitting the model...")
            clf.fit(X_train, y_train)
        st.success("Hyperparameter optimization completed!")
        best_params = clf.best_params_

        st.write('Best estimator:')

        rows = []
        for param in best_params:
            rows.append([param, best_params[param]])
        st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))

        return clf.best_estimator_

def ridge_regression(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")

    if do_hyperparameter_optimization:
        hyperparameter_optimization(X_train, y_train)

    alpha = st.number_input("Alpha", 0.0, 100.0, 1.0, key="rr_alpha")
    fit_intercept = st.checkbox("Fit Intercept", True, key="rr_fit_intercept")
    normalize = st.checkbox("Normalize", False, key="rr_normalize")
    max_iter = st.number_input("Max Iterations", 1, 100000, 1000, key="rr_max_iter")
    solver = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], index=0, key="rr_solver")


    try:
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter,
                      solver=solver)
    except ValueError as e:
        print(f"Error creating Ridge model: {e}")
        model = None

    return model
