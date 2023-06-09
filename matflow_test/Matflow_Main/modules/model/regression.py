import streamlit as st
import pickle
from modules.utils import split_xy
from modules.regressor import linear_regression, ridge_regression, lasso_regression, decision_tree_regression, random_forest_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def regression(split_name, models):
    try:
        dataset=st.session_state.splitted_data[split_name]
        train_name=dataset['train_name']
        test_name=dataset['test_name']
        train_data = st.session_state.dataset.get_data(dataset['train_name'])
        test_data = st.session_state.dataset.get_data(dataset['test_name'])
        target_var=dataset["target_var"]
        X_train, y_train = split_xy(train_data, target_var)
        X_test, y_test = split_xy(test_data, target_var)
    except:
        st.header("Properly Split Dataset First")
        return

    if "has_models"  not in st.session_state:
        st.session_state.has_models={}

    try:
        X_train, X_test = X_train.drop(target_var, axis=1), X_test.drop(target_var, axis=1)
    except:
        pass

    regressors = {
        "Linear Regression": "LR",
        "Ridge Regression": "Ridge",
        "Lasso Regression": "Lasso",
        "Decision Tree Regression": "DT",
        "Random Forest Regression": "RF"
    }
    metric_list = ["R-Squared", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error"]
    st.markdown("#")
    col1, col2 = st.columns([6.66, 3.33])
    regressor = col1.selectbox(
            "Regressor",
            regressors.keys(),
            key="model_regressor"
        )

    model_name = col2.text_input(
            "Model Name",
            f"{regressors[regressor]}_Regression",
            key="model_name"
        )

    if regressor == "Linear Regression":
        model = linear_regression.linear_regression(X_train, y_train)
    elif regressor == "Ridge Regression":
        model = ridge_regression.ridge_regression(X_train, y_train)
    elif regressor == "Lasso Regression":
        model = lasso_regression.lasso_regression(X_train, y_train)
    elif regressor == "Decision Tree Regression":
        model = decision_tree_regression.decision_tree_regressor(X_train, y_train)
    elif regressor == "Random Forest Regression":
        model = random_forest_regression.random_forest_regressor(X_train, y_train)

    col1, col2 = st.columns([6.66, 3.33])
    metrics = col1.multiselect(
            "Display Metrics",
            metric_list,
            metric_list,
            key="model_reg_metrics"
        )

    if st.button("Submit", key="model_submit"):
        if model_name not in models.list_name():
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                st.error(e)
                return
            if 'all_models' not in st.session_state:
                st.session_state.all_models = {}

            all_models = st.session_state.all_models

            selected_metrics = get_result(model, X_test, y_test, metrics)
            for met in metrics:
                st.metric(met, selected_metrics.get(met))

            result = []
            for X, y in zip([X_train, X_test], [y_train, y_test]):
                temp = get_result(model, X, y, metrics)
                result += list(temp.values())
            models.add_model(model_name, model, train_name, test_name, target_var, result,"regression")

            if split_name not in st.session_state.has_models.keys():
                st.session_state.has_models[split_name]=[]
            st.session_state.has_models[split_name].append(model_name)

            all_models.update({model_name:('Regression',dataset)})

        else:
            st.warning("Model name already exist!")

def get_result(model, X, y, metrics):
    y_pred = model.predict(X)
    metric_dict = {
        "R-Squared": r2_score(y, y_pred),
        "Mean Absolute Error": mean_absolute_error(y, y_pred),
        "Mean Squared Error": mean_squared_error(y, y_pred),
        "Root Mean Squared Error": mean_squared_error(y, y_pred, squared=False)
    }
    result=metric_dict

    return result
