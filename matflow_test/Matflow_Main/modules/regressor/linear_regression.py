import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)

    st.write('#')

    if st.button('Run Optimization'):

        st.write('#')
        param_grid = {
            "fit_intercept": [True, False],
        }
        model = LinearRegression()

        with st.spinner('Doing hyperparameter optimization...'):
            clf = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=-1)
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

def linear_regression(X_train,y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    if do_hyperparameter_optimization:
        hyperparameter_optimization(X_train, y_train)
    fit_intercept = st.checkbox("Fit Intercept", True, key="lr_fit_intercept")
    normalize = st.checkbox("Normalize", False, key="lr_normalize")
    n_jobs = st.number_input("Number of Jobs", -1, 100, -1, key="lr_n_jobs")

    try:
        model = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
    except ValueError as e:
        st.error(f"Error: {e}")
        model = None

    return model