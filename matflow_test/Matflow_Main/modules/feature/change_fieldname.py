import pandas as pd
from django.http import JsonResponse


def change_field_name(file,opt):
    temp_name=''
    print(file)
    n_iter=file.get("number_of_columns")
    selected = []
    var=[]
    var2=[]
    temp_file=file.get("data")
    for i in range(n_iter):
        var.append(temp_file[i].get("column_name"))
        selected.append(var)
        var2.append(temp_file[i].get('new_field_name'))

    modified_data=pd.DataFrame(file.get("file"))

    for i in range(n_iter):
        modified_data = modified_data.rename(columns={var[i]: var2[i]})

    df = modified_data.to_dict(orient="records")
    return JsonResponse(df, safe=False)


