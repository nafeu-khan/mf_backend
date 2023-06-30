
import pandas as pd
from django.http import JsonResponse


def merge_df(file):
    # merge_name = file.get('select_dataset_you_wanna_merge_with')
    # file_name = file.get('new_dataset_name')
    how =file.get('how')
    left_on = file.get("left_dataframe")
    right_on = file.get("right_dataframe")
    data =file.get("file")
    dataset=file.get("file2")
    tmp = pd.DataFrame(data)
    dataset=pd.DataFrame(dataset)

    temp2 = tmp.merge(dataset, left_on=left_on, right_on=right_on, how=how)

    new_value = temp2.to_dict(orient="records")
    return JsonResponse(new_value, safe=False)

