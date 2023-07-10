"""
URL configuration for Matflow project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from matflow_test.views import login, signup, display_group, test_page, display_correlation, eda_barplot, \
    display_correlation_featurePair, eda_pieplot, eda_boxplot, eda_countplot, eda_histogram, eda_violinplot, \
    eda_scatterplot, eda_regplot, eda_lineplot, eda_customplot, feature_creation, Alter_field, changeDtype, \
    merge_dataset, Encoding, Scaling, Drop_row, Drop_column, Append, Cluster, Split, Time_series, Time_series_analysis, \
    Hyper_opti, Build_model, model_evaluation, Reverse_ml, model_prediction, display_correlation_heatmap

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/signup/', signup, name='signup'),
    path('api/login/', login, name='login'),
    path('',test_page),
    path('api/display_group/', display_group, name='display-api'),
    path('api/display_correlation/', display_correlation, name='display-api'),
    path('api/display_correlation_featurePair/', display_correlation_featurePair, name='display-api'),
    path('api/display_correlation_heatmap/', display_correlation_heatmap, name='display_correlation_heatmap'),
    path('api/eda_barplot/', eda_barplot, name='bar-api'),
    path('api/eda_countplot/', eda_countplot, name='count-api'),
    path('api/eda_boxplot/', eda_boxplot, name='eda-api'),
    path('api/eda_pieplot/', eda_pieplot, name='eda-api'),
    path('api/eda_histogram/', eda_histogram, name='eda-api'),
    path('api/eda_violinplot/', eda_violinplot, name='eda-api'),
    path('api/eda_scatterplot/', eda_scatterplot, name='eda-api'),
    path('api/eda_regplot/', eda_regplot, name='reg-api'),
    path('api/eda_lineplot/', eda_lineplot, name='line-api'),
    path('api/eda_customplot/', eda_customplot, name='customplot-api'),
    path('api/feature_creation/', feature_creation, name='feature_creationapi'),
    path('api/change_dtype/',changeDtype, name='Change_dtype'),
    path('api/alter_field_name/', Alter_field, name='alter_field'),
    path('api/merge_dataset/', merge_dataset, name='merge_dataset'),
    path('api/encoding/', Encoding, name='encoding'),
    path('api/scaling/', Scaling, name='scaling'),
    path('api/drop_column/', Drop_column, name='dropping'),
    path('api/drop_rows/', Drop_row, name='dropping'),
    path('api/append/', Append, name='dropping'),
    path('api/cluster/', Cluster, name='dropping'),
    path('api/split_dataset/', Split, name='dropping'),
    path('api/build_model/', Build_model, name='dropping'),
    path('api/time_series/', Time_series, name='dropping'),
    path('api/time_series_analysis/', Time_series_analysis, name='dropping'),
    path('api/hyperparameter_optimization/', Hyper_opti, name='dropping'),
    path('api/model_evaluation/', model_evaluation, name='dropping'),
    path('api/model_prediction/', model_prediction, name='dropping'),
    path('api/reverseml/', Reverse_ml, name='dropping'),
]