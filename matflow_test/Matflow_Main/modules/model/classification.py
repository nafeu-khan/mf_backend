import streamlit as st
import pandas as pd
import pickle
from modules.utils import split_xy
from modules.classifier import knn, svm, log_reg, decision_tree, random_forest, perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def clear_opti_results_df():
    try:
       del st.session_state["opti_results_df"]
    except:
        pass


def classification(split_name, models):
    try:
        dataset = st.session_state.splitted_data[split_name]

        train_name = dataset['train_name']
        test_name = dataset['test_name']
        train_data = st.session_state.dataset.get_data(dataset['train_name'])
        test_data = st.session_state.dataset.get_data(dataset['test_name'])
        target_var = dataset["target_var"]
        X_train, y_train = split_xy(train_data, target_var)
        X_test, y_test = split_xy(test_data, target_var)
    except:
        st.header("Properly Split Dataset First")
        return

    if "has_models" not in st.session_state:
        st.session_state.has_models = {}

    try:
        X_train, X_test = X_train.drop(target_var, axis=1), X_test.drop(target_var, axis=1)
    except:
        pass

    classifiers = {
        "K-Nearest Neighbors": "KNN_Classification",
        "Support Vector Machine": "SVM_Classification",
        "Logistic Regression": "LR_Classification",
        "Decision Tree Classification": "DT_Classification",
        "Random Forest Classification": "RF_Classification",
        "Multilayer Perceptron": "MLP_Classification"
    }
    metric_list = ["Accuracy", "Precision", "Recall", "F1-Score"]

    st.markdown("#")
    col1, col2 = st.columns([6.66, 3.33])
    classifier = col1.selectbox(
        "Classifier",
        classifiers.keys(),
        key="model_classifier",
        on_change=clear_opti_results_df
    )

    model_name = col2.text_input(
        "Model Name",
        classifiers[classifier],
        key="model_name"
    )

    if classifier == "K-Nearest Neighbors":
        model = knn.knn(X_train,y_train)
    elif classifier == "Support Vector Machine":
        model = svm.svm(X_train, y_train)
    elif classifier == "Logistic Regression":
        model = log_reg.log_reg(X_train, y_train)
    elif classifier == "Decision Tree Classification":
        model = decision_tree.decision_tree(X_train, y_train)
    elif classifier == "Random Forest Classification":
        model = random_forest.random_forest(X_train, y_train)
    elif classifier == "Multilayer Perceptron":
        model = perceptron.perceptron(X_train, y_train)

    col1, col2 = st.columns([6.66, 3.33])
    metrics = col1.multiselect(
        "Display Metrics",
        metric_list,
        metric_list,
        key="model_clf_metrics"
    )

    target_nunique = y_train.nunique()
    if target_nunique > 2:
        multi_average = col2.selectbox(
            "Multiclass Average",
            ["micro", "macro", "weighted"],
            key="clf_multi_average"
        )
    else:
        multi_average = "binary"

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

            selected_metrics = get_result(model, X_test, y_test, metrics, multi_average)

            for met in metrics:
                st.metric(met, selected_metrics.get(met))

            result = []
            for X, y in zip([X_train, X_test], [y_train, y_test]):
                temp = get_result(model, X, y, metrics, multi_average)
                result += list(temp.values())

            models.add_model(model_name, model, train_name, test_name, target_var, result, "classification")

            if split_name not in st.session_state.has_models.keys():
                st.session_state.has_models[split_name] = []
            st.session_state.has_models[split_name].append(model_name)

            all_models.update({model_name: ('Classification', dataset)})

        else:
            st.warning("Model name already exist!")


def get_result(model, X, y, metrics, multi_average, pos_label=None):
    y_pred = model.predict(X)

    metric_dict = {}

    try:
        pos_label = y[1]
    except:
        pass

    # calculate accuracy
    try:
        metric_dict["Accuracy"] = accuracy_score(y, y_pred)
    except ValueError:
        metric_dict["Precision"] = "-"

        # calculate precision
    try:
        metric_dict["Precision"] = precision_score(y, y_pred, average=multi_average, pos_label=pos_label)
    except ValueError:
        metric_dict["Precision"] = "-"

    # calculate recall
    try:
        metric_dict["Recall"] = recall_score(y, y_pred, average=multi_average, pos_label=pos_label)
    except ValueError:
        metric_dict["Recall"] = "-"

    # calculate F1-score
    try:
        metric_dict["F1-Score"] = f1_score(y, y_pred, average=multi_average, pos_label=pos_label)
    except ValueError:
        metric_dict["F1-Score"] = "-"

    return metric_dict
