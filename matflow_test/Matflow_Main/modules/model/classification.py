
from django.http import JsonResponse

from ...modules.utils import split_xy
from ...modules.classifier import knn, svm, log_reg, decision_tree, random_forest, perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification(file):
    # train_name = dataset['train_name']
    # test_name = dataset['test_name']
    train_data = file.get("train_name")
    test_data = file.get("test_name")
    target_var = file.get("target_var")
    X_train, y_train = split_xy(train_data, target_var)
    X_test, y_test = split_xy(test_data, target_var)


    try:
        X_train, X_test = X_train.drop(target_var, axis=1), X_test.drop(target_var, axis=1)
    except:
        pass

    classifier = file.get("model_classifier")

    # model_name = col2.text_input(
    #     "Model Name",
    #     classifiers[classifier],
    #     key="model_name"
    # )

    if classifier == "K-Nearest Neighbors":
        model = knn.knn(X_train,y_train,file)
    elif classifier == "Support Vector Machine":
        model = svm.svm(X_train, y_train,file)
    elif classifier == "Logistic Regression":
        model = log_reg.log_reg(X_train, y_train,file)
    elif classifier == "Decision Tree Classification":
        model = decision_tree.decision_tree(X_train, y_train,file)
    elif classifier == "Random Forest Classification":
        model = random_forest.random_forest(X_train, y_train,file)
    elif classifier == "Multilayer Perceptron":
        model = perceptron.perceptron(X_train, y_train,file)

    metrics = file.get("metric_list")

    target_nunique = y_train.nunique()
    if target_nunique > 2:
        multi_average = file.get("multi_average")
    else:
        multi_average = "binary"

    model.fit(X_train, y_train)

    selected_metrics = get_result(model, X_test, y_test, metrics, multi_average)
    return JsonResponse(selected_metrics)
    # for met in metrics:
    #     st.metric(met, selected_metrics.get(met))



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
