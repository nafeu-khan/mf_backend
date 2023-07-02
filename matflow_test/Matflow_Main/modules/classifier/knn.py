import pandas as pd
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train,file):
    do_hyperparameter_optimization =file.get("do_hyperparameter_optimization ")
    if do_hyperparameter_optimization:
        n_iter = file.get("number_of_iterations_for_hyperparameter_search")
        cv = file.get("number_of_cross-validation_folds")
        random_state = file.get("random_state_for_hyperparameter_search")

        param_dist = {
            "n_neighbors": [3, 5, 10, 15, 20, 25],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "euclidean", "manhattan"]
        }
        model = KNeighborsClassifier()

        # with st.spinner('Doing hyperparameter optimization...'):

        clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                                 random_state=random_state)
        # st.spinner("Fitting the model...")
        clf.fit(X_train, y_train)

        # st.success("Hyperparameter optimization completed!")
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
        return best_param
    else :
        pass


def knn(X_train, y_train,file):
    # best_param = hyperparameter_optimization(X_train, y_train,file)
    #("Model Settings")
    n_neighbors = file.get("number_of_neighbors")
    weights = file.get("knn_weight")
    metric =file.get( "distance_metric")

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    return model
