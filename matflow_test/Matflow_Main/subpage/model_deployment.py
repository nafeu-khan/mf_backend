import pandas as pd
import streamlit as st
from django.http import JsonResponse


def model_deployment(file):
    model_name = file.get("model_name")
    # dataset = pd.DataFrame(file.get("model_name"))

    model = file.get("models")
    # train_data_name = pd.DataFrame(file.get('train_name'))
    # test_data_name=dataset['test_name']
    train_data = pd.DataFrame(file.get('train'))
    # test_data = pd.DataFrame(file.get('test'))
    target_var = file.get('target_var')

    #
    #
    col_names_all = []
    for i in train_data.columns:
        if i == target_var:
            continue
        col_names_all.append(i)
    # rad=1
    # if rad == 'All Columns':
    #     col_names = col_names_all
    # else:
    #     col_names = st.multiselect('Custom Columns', col_names_all, help='Other values will be 0 as default value')
    col_names= file.get("col_names")
    prediction = ['']
    correlations = train_data[col_names + [target_var]].corr()[target_var]

    result=[]
    for i in col_names:
        threshold=train_data[i].abs().max()
        result[i]=threshold
    X = [result[i] if i in col_names else 0 for i in col_names_all]
    prediction=model.predict(X)


    obj={
          'pred' : prediction[0],
        }

    return JsonResponse (obj)