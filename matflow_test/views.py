import json

import pandas as pd
import numpy as np
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from io import BytesIO
import matplotlib.pyplot as plt
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login
from rest_framework import status
from django.contrib.auth.models import User

@api_view(['POST'])
def signup(request):
    try:
        username = request.data.get('username')
        password = request.data.get('password')
    except:
        return Response({'error': 'Please provide both username and password.'}, status=status.HTTP_400_BAD_REQUEST)

    if not username or not password:
        return Response({'error': 'Please provide both username and password.'}, status=status.HTTP_400_BAD_REQUEST)

    # Check if the username already exists
    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already exists.'}, status=status.HTTP_400_BAD_REQUEST)

    # Create a new user
    user = User.objects.create_user(username=username, password=password)

    return Response({'message': 'User created successfully.'}, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def login(request):
    try:
        username = request.data.get('username')
        password = request.data.get('password')
    except:
        return Response({'error': 'Please provide both username and password.'}, status=status.HTTP_400_BAD_REQUEST)

    user = authenticate(username=username, password=password)

    if not user:
        return Response({'error': 'Invalid credentials.'}, status=status.HTTP_401_UNAUTHORIZED)

    # Log in the user
    login(request, user)

    return Response({'message': 'User logged in successfully.'}, status=status.HTTP_200_OK)


def test_page(request):

    return render(request, 'index.html')

from .Matflow_Main.modules import utils


@api_view(['GET', 'POST'])
def display_group(request):
    data = json.loads(request.body)
    # data = request.FILES.get('file')
    file = data.get('file')
    # file=pd.read_csv(file)
    file = pd.DataFrame(file)

    group_var = data.get("group_var")

    agg_func = data.get("agg_func")

    data = file.groupby(by=group_var, as_index=False).agg(agg_func)
    data = data.to_json(orient='records')
    return JsonResponse({'data': data})
@api_view(['GET','POST'])
def display(request):
    print(request)
    data = request.FILES.get('file')
    data = request.body.get('file')
    data=pd.read_csv(data)
    group_var = request.body.get("group_var")

    agg_func = request.body.get("filter_agg_func")

    data.groupby(by=group_var, as_index=False).agg(agg_func)

    return JsonResponse({'data':data})


def custom(data, var, params):
    idx_start = int(params.get("idx_start", 0))
    idx_end = int(params.get("idx_end", data.shape[0]))
    is_filter = params.get("is_filter", False)

    if is_filter:
        filtered_data = filter_data(data, params, var)
        data_slice = filtered_data.loc[idx_start:idx_end, var]
    else:
        data_slice = data.loc[idx_start:idx_end, var]

    return data_slice.to_dict(orient="records")


def filter_data(data, params, display_var):
    filter_var = params.get("filter_var", "")
    filter_operator = params.get("filter_cond", "")
    filter_value = params.get("filter_value", "")

    filtered_data = filter_result(data, filter_var, filter_operator, filter_value)
    result = filtered_data[display_var]

    return result


def filter_result(data, filter_var, filter_operator, filter_value):
    if filter_operator == "<":
        result = data.loc[data[filter_var] < filter_value]
    elif filter_operator == ">":
        result = data.loc[data[filter_var] > filter_value]
    elif filter_operator == "==":
        if type(filter_value) != str:  # np.isna() cannot pass str as parameter
            if np.isnan(filter_value):  # check if value is nan
                result = data.loc[data[filter_var].isna() == True]
            else:
                result = data.loc[data[filter_var] == filter_value]
        else:
            result = data.loc[data[filter_var] == filter_value]
    elif filter_operator == "<=":
        result = data.loc[data[filter_var] <= filter_value]
    elif filter_operator == ">=":
        result = data.loc[data[filter_var] >= filter_value]
    else:
        if type(filter_value) != str:  # np.isna() cannot pass str as parameter
            if np.isnan(filter_value):  # check if value is nan
                result = data.loc[data[filter_var].isna() == False]
            else:
                result = data.loc[data[filter_var] == filter_value]
        else:
            result = data.loc[data[filter_var] != filter_value]

    return result
