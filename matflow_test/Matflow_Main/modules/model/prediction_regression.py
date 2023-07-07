import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import JsonResponse
from statsmodels.graphics.gofplots import qqplot
import plotly.graph_objects as go
import plotly.io as pio
from ...modules import utils
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def prediction_regression(dataset, models,model_opt,file):

    target_var = file.get( "Target Variable")
    data = file.get("file")
    X, y = utils.split_xy(data, target_var)
    y_pred = file.get("y_pred")

    result_opt = file.get("Result")
    show_result(y, y_pred, result_opt)

def show_result(y, y_pred, result_opt):
    if result_opt == "Target Value":
        graph_header = st.text_input("Enter graph header", "Actual vs. Predicted Values")

        result = pd.DataFrame({
            "Actual": y,
            "Predicted": y_pred
        })
        result=result.to_json(orient="records")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predicted'))
        fig.update_layout(
            title=graph_header,
            xaxis=dict(title='Index'),
            yaxis=dict(title='Value')
        )
        graph_json = fig.to_json()
        obj= {
            "table": result,
            "graph":graph_json
        }
        return JsonResponse (obj)

    elif result_opt == "R2 Score":
        result = r2_score(y, y_pred)
        st.metric(result_opt, result)

    elif result_opt == "MAE":
        result = mean_absolute_error(y, y_pred)
        st.metric(result_opt, result)

    elif result_opt == "MSE":
        result = mean_squared_error(y, y_pred)
        st.metric(result_opt, result)

    elif result_opt == "RMSE":
        result = np.sqrt(mean_squared_error(y, y_pred))
        st.metric(result_opt, result)

    elif result_opt == "Actual vs. Predicted":
        actualvspred(y,y_pred,graph_header="Actual vs. Predicted Values")

    elif result_opt == "Residuals vs. Predicted":

        residuals = y - y_pred

        fig, ax = plt.subplots(figsize=(8, 6))

        ax = sns.scatterplot(x=y_pred, y=residuals)

        ax.axhline(y=0, color="red", linestyle="--")

        ax.set_xlabel("Predicted")

        ax.set_ylabel("Residuals")

        ax.set_title("Residuals vs. Predicted Values")

        st.pyplot(fig)


    elif result_opt == "Histogram of Residuals":

        residuals = y - y_pred

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.histplot(residuals, kde=True, ax=ax)

        ax.set_xlabel("Residuals")

        ax.set_ylabel("Frequency")

        ax.set_title("Histogram of Residuals")

        st.pyplot(fig)


    elif result_opt == "QQ Plot":

        residuals = y - y_pred

        fig, ax = plt.subplots(figsize=(8, 6))

        qqplot(residuals, line='s', ax=ax)

        ax.set_xlabel("Theoretical Quantiles")

        ax.set_ylabel("Sample Quantiles")

        ax.set_title("Normal Q-Q Plot of Residuals")

        st.pyplot(fig)


    elif result_opt == "Box Plot of Residuals":

        residuals = y - y_pred

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(residuals, ax=ax)

        ax.set_xlabel("Residuals")

        ax.set_title("Box Plot of Residuals")

        st.pyplot(fig)
    elif result_opt == "Regression Line Plot":
        fig, ax = plt.subplots(figsize=(8, 6))

        ax = sns.regplot(x=y, y=y_pred, label="Prediction")

        ax.set_xlabel("Actual")

        ax.set_ylabel("Predicted")

        ax.set_title("Regression Line Plot")

        st.pyplot(fig)

def actualvspred(y,y_pred,graph_header):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax = sns.lineplot(x=range(len(y)), y=y, label="Actual")

    ax = sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")

    ax.set_xlabel("Index")

    ax.set_ylabel("Value")

    ax.set_title(graph_header)

    st.pyplot(fig)
