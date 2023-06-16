import json
import pandas as pd
import numpy as np
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from rest_framework import status
from django.contrib.auth.models import User
from .Matflow_Main.modules.dataframe.correlation import display_pair
from .Matflow_Main.modules.feature.creation import creation
from .Matflow_Main.modules.graph.barplot import Barplot
from .Matflow_Main.modules.graph.customplot import comparison_plot
from .Matflow_Main.modules.graph.lineplot import Lineplot
from .Matflow_Main.modules.graph.pieplot import Pieplot
from .Matflow_Main.modules.graph.countplot import Countplot
from .Matflow_Main.modules.graph.boxplot import Boxplot
from .Matflow_Main.modules.graph.histogram import Histogram
from .Matflow_Main.modules.graph.regplot import Regplot
from .Matflow_Main.modules.graph.scatterplot import Scatterplot
from .Matflow_Main.modules.graph.violinplot import Violinplot



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
@api_view(['GET', 'POST'])
def display_group(request):
    data = json.loads(request.body)
    file = data.get('file')
    # file=pd.read_csv(file)
    file = pd.DataFrame(file)

    group_var = data.get("group_var")

    agg_func = data.get("agg_func")

    numeric_columns = file.select_dtypes(include='number').columns

    data = file.groupby(by=group_var, as_index=False).agg(agg_func)
    data = data.to_json(orient='records')
    return JsonResponse({'data': data})
@api_view(['GET', 'POST'])
def display_correlation(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    correlation_method="kendall"
    file = file.select_dtypes(include='number')
    correlation_data =file.corr(correlation_method)
    data = correlation_data.to_json(orient='records')
    return JsonResponse({'data': data})

@api_view(['GET','POST'])
def display_correlation_featurePair(request):
    data = json.loads(request.body)
    correlation_data =pd.DataFrame(data.get('file'))
    bg_gradient= data.get('gradient')
    feature1 = data.get('feature1')
    feature2 = data.get('feature2')
    drop = data.get('drop')
    absol = data.get('absol')
    high = data.get('high')
    df=display_pair(correlation_data,bg_gradient,feature1,feature2,high,drop,absol)
    data = df.to_json(orient='records')
    return JsonResponse({'data': data})

@api_view(['GET','POST'])
def eda_barplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    cat = data.get('cat')  # Get the categorical variable from the query parameter
    num = data.get('num')  # Get the numerical variable from the query parameter
    hue = data.get('hue')  # Get the hue variable from the query parameter
    orient = data.get('orient')  # Get the orientation from the query parameter
    annote=data.get('annote')
    title=data.get('title')

    response= Barplot(file,cat,num,hue,orient,annote,title)
    return response
@api_view(['GET','POST'])
def eda_pieplot(request):
    data =json.loads((request.body))
    file=data.get('file')
    file = pd.DataFrame(file)
    var=data.get('cat')
    explode=data.get('gap')
    title=data.get('title')
    label=data.get('label')
    percentage=data.get('percentage')
    return Pieplot(file,var,explode,title,label,percentage)
@api_view(['GET','POST'])
def eda_countplot(request):
    data =json.loads((request.body))
    file=data.get('file')
    file = pd.DataFrame(file)
    var=data.get('cat')
    title=data.get('title')
    hue=data.get('hue')
    orient=data.get('orient')
    annotate=data.get('annote')
    return Countplot(file,var,title,hue,orient,annotate)
@api_view(['GET','POST'])
def eda_boxplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    cat = data.get('cat')  # Get the categorical variable from the query parameter
    num = data.get('num')  # Get the numerical variable from the query parameter
    hue = data.get('hue')  # Get the hue variable from the query parameter
    orient = data.get('orient')  # Get the orientation from the query parameter
    title = data.get('title')
    dodge=data.get('dodge')
    response= Boxplot(file,title,cat,num,hue,orient,dodge)
    return response
@api_view(['GET','POST'])
def eda_histogram(request):
    data =json.loads((request.body))
    file=data.get('file')
    file = pd.DataFrame(file)
    var=data.get('var')
    title=data.get('title')
    hue=data.get('hue')
    orient=data.get('orient')
    agg=data.get('agg')
    autoBin=data.get('autoBin')
    kde=data.get('kde')
    legend=data.get('legend')
    return Histogram(file,var,title,hue,orient,agg,autoBin,kde,legend)
@api_view(['GET','POST'])
def eda_violinplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    cat = data.get('cat')  # Get the categorical variable from the query parameter
    num = data.get('num')  # Get the numerical variable from the query parameter
    hue = data.get('hue')  # Get the hue variable from the query parameter
    orient = data.get('orient')  # Get the orientation from the query parameter
    dodge=data.get('dodge')
    split=data.get('split')

    title=data.get('title')
    response= Violinplot(file,cat,num,hue,orient,dodge,split,title)
    return response
@api_view(['GET','POST'])
def eda_scatterplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    x_var = data.get('x_var')
    y_var = data.get('y_var')
    title = data.get('title')
    hue = data.get('hue')
    response= Scatterplot(file,x_var,y_var,hue,title)
    return response
@api_view(['GET','POST'])
def eda_regplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    x_var = data.get('x_var')
    y_var = data.get('y_var')
    title = data.get('title')
    sctr = data.get('scatter')
    response= Regplot(file,x_var,y_var,title,sctr)
    return response

@api_view(['GET','POST'])
def eda_lineplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    x_var = data.get('x_var')
    y_var = data.get('y_var')
    title = data.get('title')
    hue = data.get('hue')
    style = data.get('style')
    legend = data.get('legend')
    response= Lineplot(file,x_var,y_var,hue,title,style,legend)
    return response
@api_view(['GET','POST'])
def eda_customplot(request):
    data = json.loads(request.body)
    file = data.get('file')
    file = pd.DataFrame(file)
    x_var = data.get('x_var')
    y_var = data.get('y_var')
    hue = data.get('hue')
    response= comparison_plot(file,x_var,y_var,hue)
    return response
@api_view(['GET','POST'])
def feature_creation(request):
    data=json.loads(request.body)
    response = creation(data)
    return response

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
