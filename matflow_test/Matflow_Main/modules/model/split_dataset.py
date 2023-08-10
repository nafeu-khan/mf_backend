import pandas as pd
from django.http import JsonResponse
from sklearn.model_selection import train_test_split

def split_dataset(file):
    data = pd.DataFrame (file.get("file"))
    target_var = file.get("target_variable")
    stratify = file.get("stratify")
    test_size = float(file.get("test_size"))
    random_state = int(file.get("random_state"))
    train_name =file.get("train_data_name")
    test_name = file.get("test_data_name")
    shuffle = file.get("split_shuffle")
    split_dataset_name=file.get("split_name")
    stratify = None if (stratify == "-") else stratify
    X = data
    y = data[target_var]
    X_train, X_test = train_test_split(X, test_size=test_size,random_state=random_state)

    X_train=X_train.to_dict(orient="records")
    X_test = X_test.to_dict(orient="records")
    obj={
        "train": X_train ,
        "test": X_test
    }

    # df = obj.to_dict(orient="records")
    return JsonResponse(obj, safe=False)