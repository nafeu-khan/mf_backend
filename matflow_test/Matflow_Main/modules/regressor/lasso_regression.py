import streamlit as st
from django.http import JsonResponse
from sklearn.linear_model import Lasso
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train,file):
    n_iter = int(file.get("Number of iterations for hyperparameter search"))
    cv = int(file.get("Number of cross-validation folds"))
    random_state = int(file.get("Random state for hyperparameter search"))
    param_dist = {
        "alpha": [0.001, 0.01, 0.1, 1, 10],
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000, 10000],
        "positive": [True, False],
        "selection": ["cyclic", "random"],
        "tol": [0.0001, 0.001, 0.01, 0.1],
        "warm_start": [True, False]
    }
    model = Lasso()

    # for i in range(100):
        # time.sleep(0.1)
        # st.spinner(f"Running iteration {i + 1} of {n_iter}...")
    clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                             random_state=random_state)
    # st.spinner("Fitting the model...")
    clf.fit(X_train, y_train)
    st.success("Hyperparameter optimization completed!")
    best_params = clf.best_params_

    st.write('Best estimator:')

    rows = []
    for param in best_params:
        rows.append([param, best_params[param]])
    st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))
    best_param = clf.best_params_
    return JsonResponse(best_param)

def lasso_regression(X_train,y_train):

    alpha = st.number_input("Alpha", 0.0, 100.0, 1.0, key="ls_alpha")
    fit_intercept = st.checkbox("Fit Intercept", True, key="ls_fit_intercept")
    normalize = st.checkbox("Normalize", False, key="ls_normalize")
    max_iter = st.number_input("Max Iterations", 1, 100000, 1000, key="ls_max_iter")
    selection = st.selectbox("Selection", ["cyclic", "random"], index=0, key="ls_selection")

    try:
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter,
                      selection=selection)
    except ValueError as e:
        # Handle any ValueErrors that occur during initialization
        print("Error: ", e)
    except TypeError as e:
        # Handle any TypeErrors that occur during initialization
        print("Error: ", e)
    except Exception as e:
        # Handle any other exceptions that occur during initialization
        print("Error: ", e)

    return model
