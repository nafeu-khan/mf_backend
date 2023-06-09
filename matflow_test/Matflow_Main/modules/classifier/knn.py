import pandas as pd
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
    if do_hyperparameter_optimization:
        st.subheader("Hyperparameter Optimization Settings")
        n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=5, step=1)
        cv = st.number_input("Number of cross-validation folds", min_value=2, value=2, step=1)
        random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    if "knn_best_param" not in st.session_state:
        st.session_state.knn_best_param={
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski"
    }


    st.write('#')

    if do_hyperparameter_optimization and st.button('Run Optimization'):

        st.write('#')
        param_dist = {
            "n_neighbors": [3, 5, 10, 15, 20, 25],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "euclidean", "manhattan"]
        }
        model = KNeighborsClassifier()

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


        if "opti_results_df" not in  st.session_state:
            st.session_state.opti_results_df=pd.DataFrame

        st.session_state.opti_results_df=results_df

        best_param = clf.best_params_

        st.session_state.knn_best_param = best_param


    return st.session_state.knn_best_param


def knn(X_train, y_train):
    best_param = hyperparameter_optimization(X_train, y_train)

    try:
        st.write('#')
        st.write(st.session_state.opti_results_df)
    except:
        pass

    st.subheader("Model Settings")

    col1, col2, col3 = st.columns(3)
    n_neighbors = col1.number_input(
        "Number of Neighbors",
        1, 100, best_param["n_neighbors"],
        key="knn_neighbors"
    )
    weights = col2.selectbox(
        "Weight Function",
        ["uniform", "distance"],
        key="knn_weight",
        index=[0, 1].index(["uniform", "distance"].index(best_param["weights"]))
    )
    metric = col3.selectbox(
        "Distance Metric",
        ["minkowski", "euclidean", "manhattan"],
        index=[0, 1, 2].index(["minkowski", "euclidean", "manhattan"].index(best_param["metric"]))
    )

    try:
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    except ValueError as e:
        print("An error occurred while creating the KNeighborsClassifier model: ", e)
    except Exception as e:
        print("An unexpected error occurred: ", e)

    return model
