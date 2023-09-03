import base64
import json
import pandas as pd
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from rest_framework import status
from django.contrib.auth.models import User

from .Matflow_Main.modules.classifier import knn, svm, log_reg, decision_tree, random_forest, perceptron
from .Matflow_Main.modules.dataframe.correlation import display_heatmap, display_pair
from .Matflow_Main.modules.feature.append import append
from .Matflow_Main.modules.feature.change_dtype import Change_dtype
from .Matflow_Main.modules.feature.change_fieldname import change_field_name
from .Matflow_Main.modules.feature.cluster import cluster_dataset
from .Matflow_Main.modules.feature.creation import creation
from .Matflow_Main.modules.feature.dropping import  drop_row, drop_column
from .Matflow_Main.modules.feature.encoding import encoding
from .Matflow_Main.modules.feature.merge_dataset import merge_df
from .Matflow_Main.modules.feature.scaling import scaling
from .Matflow_Main.modules.graph.barplot import Barplot
from .Matflow_Main.modules.graph.customplot import Custom_plot
from .Matflow_Main.modules.graph.lineplot import Lineplot
from .Matflow_Main.modules.graph.pieplot import Pieplot
from .Matflow_Main.modules.graph.countplot import Countplot
from .Matflow_Main.modules.graph.boxplot import Boxplot
from .Matflow_Main.modules.graph.histogram import Histogram
from .Matflow_Main.modules.graph.regplot import Regplot
from .Matflow_Main.modules.graph.scatterplot import Scatterplot
from .Matflow_Main.modules.graph.violinplot import Violinplot
from .Matflow_Main.modules.model.classification import classification
from .Matflow_Main.modules.model.model_report import model_report
from .Matflow_Main.modules.model.prediction_classification import prediction_classification
from .Matflow_Main.modules.model.prediction_regression import prediction_regression
from .Matflow_Main.modules.model.regression import regression
from .Matflow_Main.modules.model.split_dataset import split_dataset
from .Matflow_Main.modules.regressor import linear_regression, ridge_regression, lasso_regression, \
    decision_tree_regression, random_forest_regression, svr
from .Matflow_Main.modules.utils import split_xy
from .Matflow_Main.subpage.Reverse_ML import reverse_ml
from .Matflow_Main.subpage.temp import temp
from .Matflow_Main.subpage.time_series import  time_series
from .Matflow_Main.subpage.time_series_analysis import  time_series_analysis


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
# @api_view(['GET', 'POST'])
# def test_page(request):
#     return HttpResponse("hello")
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
def display_correlation_heatmap(request):
    data = json.loads(request.body)
    correlation_data =pd.DataFrame(data.get('file'))
    response= display_heatmap(correlation_data)
    return response
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
    response= Custom_plot(file,x_var,y_var,hue)
    return response
@api_view(['GET','POST'])
def feature_creation(request):
    data=json.loads(request.body)
    response = creation(data)
    return response
@api_view(['GET','POST'])
def changeDtype(request):
    data=json.loads(request.body)
    response = Change_dtype(data)
    return response
@api_view(['GET','POST'])
def Alter_field(request):
    data=json.loads(request.body)
    response = change_field_name(data)
    return response
@api_view(['GET','POST'])
def merge_dataset(request):
    data=json.loads(request.body)
    response = merge_df(data)
    return response
@api_view(['GET','POST'])
def Encoding(request):
    data=json.loads(request.body)
    response = encoding(data)
    return response
@api_view(['GET','POST'])
def Scaling(request):
    data=json.loads(request.body)
    response = scaling(data)
    return response
@api_view(['GET','POST'])
def Drop_column(request):
    data=json.loads(request.body)
    response = drop_column(data)
    return response
@api_view(['GET','POST'])
def Drop_row(request):
    data=json.loads(request.body)
    response = drop_row(data)
    return response
@api_view(['GET','POST'])
def Append(request):
    data=json.loads(request.body)
    response = append(data)
    return response
@api_view(['GET','POST'])
def Cluster(request):
    data=json.loads(request.body)
    response = cluster_dataset(data)
    return response
@api_view(['GET','POST'])
def Split(request):
    data=json.loads(request.body)
    response = split_dataset(data)
    print(response)
    return response
@api_view(['GET','POST'])
def Build_model(request):
    data=json.loads(request.body)
    response = split_dataset(data)
    return response
@api_view(['GET','POST'])
def Hyper_opti(request):
    data=json.loads(request.body)
    print(data.keys())
    train_data=pd.DataFrame(data.get("train"))
    test_data=pd.DataFrame(data.get("test"))
    target_var=data.get("target_var")
    print(target_var)
    # print(f"{train_data.head} {test_data.head} {target_var}")
    X_train, y_train = split_xy(train_data, target_var)
    X_test, y_test = split_xy(test_data, target_var)
    type=data.get("type")
    if(type=="classifier"):
        classifier=data.get("classifier")
        if(classifier=="K-Nearest Neighbors"):
            response= knn.hyperparameter_optimization(X_train, y_train,data)
        elif(classifier=="Support Vector Machine"):
            response= svm.hyperparameter_optimization(X_train, y_train,data)
        elif(classifier=="Logistic Regression"):
            response= log_reg.hyperparameter_optimization(X_train, y_train,data)
        elif(classifier=="Decision Tree Classification"):
            response= decision_tree.hyperparameter_optimization(X_train, y_train,data)
        elif(classifier=="Random Forest Classification"):
            response = random_forest.hyperparameter_optimization(X_train, y_train, data)
        elif(classifier=="Multilayer Perceptron"):
            response = perceptron.hyperparameter_optimization(X_train, y_train, data)
    else :
        regressor = data.get("regressor")
        if regressor == "Linear Regression":
            response = linear_regression.hyperparameter_optimization(X_train, y_train,data)
        elif regressor == "Ridge Regression":
            response = ridge_regression.hyperparameter_optimization(X_train, y_train,data)
        elif regressor == "Lasso Regression":
            response = lasso_regression.hyperparameter_optimization(X_train, y_train,data)
        elif regressor == "Decision Tree Regression":
            response = decision_tree_regression.hyperparameter_optimization(X_train, y_train,data)
        elif regressor == "Random Forest Regression":
            response = random_forest_regression.hyperparameter_optimization(X_train, y_train,data)
        elif regressor == "Support Vector Regressor":
            print("in svr")
            response = svr.hyperparameter_optimization(X_train, y_train,data)
    return response
@api_view(['GET','POST'])
def Build_model(request):
    data=json.loads(request.body)
    type=data.get("type")
    if(type== "classifier"):
        response = classification(data)
    else:
        response = regression(data)
    return response
@api_view(['GET','POST'])
def model_evaluation(request):
    data=json.loads(request.body)
    response = model_report(data)
    return response
@api_view(['GET','POST'])
def model_prediction(request):
    data=json.loads(request.body)
    type=data.get("type")
    print(type)
    if(type=="regressor"):
        response=prediction_regression(data)
    else:
        response = prediction_classification(data)
    return response
import pickle
from django.http import HttpResponse
@api_view(['GET','POST'])
def download_model(file):
    model = pickle.loads(file.get("model"))
    model_binary = pickle.dumps(model)
    response = HttpResponse(model_binary, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="model_name".pkl"'
    return response

@api_view(['GET', 'POST'])
def deploy_data(request):
    file = json.loads(request.body)
    train_data = pd.DataFrame(file.get('train'))
    target_var = file.get('target_var')
    col_names_all = [col for col in train_data.columns if col != target_var]
    col_names = train_data.columns.tolist()
    # correlations = train_data[col_names_all + [target_var]].corr()[target_var]
    correlations = train_data[col_names_all + [target_var]].corr(numeric_only=True)[target_var]
    result = []
    for col in col_names_all:
        threshold = train_data[col].abs().max()
        data_type = 'int' if np.issubdtype(train_data[col].dtype, np.integer) else 'float'
        threshold = float(threshold) if correlations[col] >= 0 else float(-threshold)
        result.append({"col": col, "value": float(threshold) if data_type == 'float' else int(threshold), "data_type": data_type})
    response = {"result": result}
    return JsonResponse(response)

@api_view(['GET','POST'])
def deploy_result(request):
    file = json.loads(request.body)
    model_bytes = base64.b64decode(file.get("model_deploy"))
    model = pickle.loads(model_bytes)
    result = file.get("result")
    train_data = pd.DataFrame(file.get('train'))
    target_var=file.get('target_var')
    print(result)
    print(target_var)
    col_names_all = []
    col_names=[]
    for i in train_data.columns:
        if i!=target_var:
            col_names_all.append(i)
    col_names.extend(result.keys())
    print(col_names)
    print(col_names_all)
    X = [result[i] if i in col_names  else 0 for i in col_names_all]
    # prediction = model.get_prediction(model_name, [X])
    print(X)

    prediction = model.predict([X])
    obj = {
        'pred': prediction[0],
    }
    return JsonResponse(obj)
@api_view(['GET','POST'])
def Time_series(request):
    data=json.loads(request.body)
    response = time_series(data)
    return response
@api_view(['GET','POST'])
def Time_series_analysis(request):
    data=json.loads(request.body)
    response = time_series_analysis(data)
    return response
@api_view(['GET','POST'])
def Reverse_ml(request):
    data=json.loads(request.body)
    response = reverse_ml(data)
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
