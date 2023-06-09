import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

from modules import utils
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def prediction(dataset, models,model_opt):
    show = False
    col1, col2, col3 = st.columns(3)

    data_opt = col1.selectbox(
        "Select Data",
        dataset.list_name()
    )

    target_var = col2.selectbox(
        "Target Variable",
        models.target_var
    )

    if st.checkbox("Show Result"):
        data = dataset.get_data(data_opt)
        X, y = utils.split_xy(data, target_var)
        y_pred = models.get_prediction(model_opt, X)

        col1, col2 = st.columns(2)
        result_opt = col1.selectbox(
            "Result",
            ["Target Value", "R2 Score", "MAE", "MSE", "RMSE", "Regression Line Plot","Actual vs. Predicted",
             "Residuals vs. Predicted", "Histogram of Residuals", "QQ Plot", "Box Plot of Residuals"]
        )

        show_result(y, y_pred, result_opt)

def show_result(y, y_pred, result_opt):


    if result_opt == "Target Value":
        c1,c2=st.columns(2)
        with c1:
            graph_header = st.text_input("Enter graph header", "Actual vs. Predicted Values")
        st.markdown("#")
        result = pd.DataFrame({
            "Actual": y,
            "Predicted": y_pred
        })
        st.markdown("#")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(result)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = sns.lineplot(x=range(len(y)), y=y, label="Actual")
            ax = sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title(graph_header)
            st.pyplot(fig)

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
