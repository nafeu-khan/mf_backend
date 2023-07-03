import pandas as pd
from django.http import JsonResponse
from ...modules.utils import split_xy
from ...modules.classifier import knn, svm, log_reg, decision_tree, random_forest, perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification(file):
    train_data = pd.DataFrame(file.get("train"))
    test_data = pd.DataFrame(file.get("test"))
    target_var = file.get("target_var")
    X_train, y_train = split_xy(train_data, target_var)
    X_test, y_test = split_xy(test_data, target_var)
    try:
        X_train, X_test = X_train.drop(target_var, axis=1), X_test.drop(target_var, axis=1)
    except:
        pass
    classifier = file.get("classifier")
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

    # metrics = file.get("metric_list")

    target_nunique = y_train.nunique()
    if target_nunique > 2:
        multi_average = file.get("Multiclass Average")
        print(f"multi = {multi_average}")
    else:
        multi_average = "binary"

    model.fit(X_train, y_train)
    selected_metrics = get_result(model, X_test, y_test, multi_average,pos_label=None)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    result = []
    i=0
    for X, y in zip([X_train, X_test], [y_train, y_test]):
        list2 = get_result(model, X, y, metrics, multi_average,pos_label=None)
        if(i==0):
            list1=get_result(model, X, y, metrics, multi_average,pos_label=None)
        i+=1

    merged_list = {
        f"Train {key}": value
        for key, value in list1.items()
    }

    merged_list.update({
        f"Test {key}": value
        for key, value in list2.items()
    })

    obj={
        "metrics": selected_metrics,   #4
        "metrics_table":merged_list     #8
    }
    return JsonResponse(obj)

def get_result(model, X, y, multi_average,pos_label=None):
    # multi_average="micro"
    y_pred = model.predict(X)
    metric_dict = {}
    metric_dict["Accuracy"] = accuracy_score(y, y_pred)
    metric_dict["Precision"] = precision_score(y, y_pred, average=multi_average)
    metric_dict["Recall"] = recall_score(y, y_pred, average=multi_average)
    metric_dict["F1-Score"] = f1_score(y, y_pred, average=multi_average)
    return metric_dict

