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
from matflow_test.views import login,signup,display_group,test_page,display_correlation,eda_barplot,display_correlation_featurePair,eda_pieplot,eda_boxplot,eda_countplot

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/signup/', signup, name='signup'),
    path('api/login/', login, name='login'),
    path('',test_page),
    path('api/display_group/', display_group, name='display-api'),
    path('api/display_correlation/', display_correlation, name='display-api'),
    path('api/display_correlation_featurePair/', display_correlation_featurePair, name='display-api'),
    path('api/eda_barplot/', eda_barplot, name='bar-api'),
    path('api/eda_countplot/', eda_countplot, name='count-api'),
    path('api/eda_boxplot/', eda_boxplot, name='box-api'),
    path('api/eda_pieplot/', eda_pieplot, name='box-api'),

]