import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modules import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score


def prediction(dataset, models,model_opt):
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
            ["Target Value", "Accuracy", "Precision", "Recall", "F1-Score",
             "Classification Report", "Confusion Matrix", "Actual vs. Predicted",
             "Precision-Recall Curve", "ROC Curve"]

        )

        # try:
        #     if y.nunique() > 2:
        #         multi_average = col2.selectbox(
        #             "Multiclass Average",
        #             ["micro", "macro", "weighted"],
        #             key="prediction_multi_average"
        #         )
        #     else:
        #         multi_average = "binary"
        #
        #     show_result(y, y_pred, result_opt, multi_average)
        #
        # except ValueError as e:
        #     st.warning(str(e))
        #
        # except TypeError as e:
        #     st.warning(str(e))
        try:
            if y.nunique() > 2:
                # multiclass case (denied)
                # show_multiclass(y,y_pred)
                show_result(y, y_pred, result_opt, None,X,model_opt)
            else:
                # binary case
                show_result(y, y_pred, result_opt, "binary",X,model_opt)
        except ValueError as e:
            st.warning(str(e))

        except TypeError as e:
            st.warning(str(e))


def show_result(y, y_pred, result_opt, multi_average,X,model_name):
    # global pred_prob
    le = LabelEncoder()

    if result_opt in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        metric_dict = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average=multi_average),
            "Recall": recall_score(y, y_pred, average=multi_average),
            "F1-Score": f1_score(y, y_pred, average=multi_average)
        }

        result = metric_dict.get(result_opt)
        st.metric(result_opt, result)

    elif result_opt == "Target Value":
        # col=st.columns(1)
        graph_header = st.text_input("Enter graph header", "Actual vs. Predicted Values")
        st.markdown("#")
        result = pd.DataFrame({
            "Actual": y,
            "Predicted": y_pred
        })
        col1,col2=st.columns(2)

        with col1:
            st.dataframe(result)
        with col2:
            actvspred(y,y_pred,graph_header)

    elif result_opt == "Classification Report":
        result = classification_report(y, y_pred)

        st.text("`" + result)

    elif result_opt == "Confusion Matrix":
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        ax = sns.heatmap(
            cm, annot=True,
            fmt='.4g',
            xticklabels=np.unique(y_pred),
            yticklabels=np.unique(y_pred)
        )

        ax.set_title(result_opt)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig)

    elif result_opt == "Actual vs. Predicted":
        actvspred(y,y_pred,graph_header="Actual vs. Predicted Values")
    elif result_opt == "Precision-Recall Curve":
        if y.nunique() > 2:
            # st.warning("Precision-Recall curve is not supported for multiclass classification")

            label_encoder = LabelEncoder()
            label_encoder.fit(y)
            y = label_encoder.transform(y)
            classes = label_encoder.classes_

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            min_max_scaler = MinMaxScaler()
            X_train_norm = min_max_scaler.fit_transform(X_train)
            X_test_norm = min_max_scaler.fit_transform(X_test)

            RF = OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
            RF.fit(X_train_norm, y_train)
            y_pred = RF.predict(X_test_norm)
            pred_prob = RF.predict_proba(X_test_norm)

            y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

            precision = {}
            recall = {}
            average_precision = dict()

            n_class = classes.shape[0]

            for i in range(n_class):
                precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], pred_prob[:, i])
                average_precision[i] = average_precision_score(y_test_binarized[:, i], pred_prob[:, i])

                plt.plot(recall[i], precision[i], linestyle='--',
                         label='%s (AP=%0.2f)' % (classes[i], average_precision[i]))

            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.title('Multiclass Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
            plt.show()
            st.pyplot(plt.gcf())

            return
        else:
            # encode y and y_pred
            y_encoded = le.fit_transform(y)
            y_pred_encoded = le.transform(y_pred)

            precision, recall, _ = precision_recall_curve(y_encoded.ravel(), y_pred_encoded.ravel())

            fig, ax = plt.subplots()

            sns.lineplot(x=recall, y=precision, ax=ax)

            ax.set_xlabel("Recall")

            ax.set_ylabel("Precision")

            ax.set_title("Precision-Recall Curve")

            st.pyplot(fig)

    elif result_opt == "ROC Curve":
        if y.nunique() > 2:
            label_encoder = LabelEncoder()
            label_encoder.fit(y)
            y = label_encoder.transform(y)
            classes = label_encoder.classes_

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            min_max_scaler = MinMaxScaler()
            X_train_norm = min_max_scaler.fit_transform(X_train)
            X_test_norm = min_max_scaler.fit_transform(X_test)

            if (model_name == "RF_Classification"):
                RF = OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
                RF.fit(X_train_norm, y_train)
                y_pred = RF.predict(X_test_norm)
                pred_prob = RF.predict_proba(X_test_norm)
            elif (model_name == "MLP_Classification"):
                mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
                mlp.fit(X_train_norm, y_train)
                y_pred = mlp.predict(X_test_norm)
                pred_prob = mlp.predict_proba(X_test_norm)
            elif (model_name == "KNN_Classification"):
                k = 5
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_norm, y_train)
                y_pred = knn.predict(X_test_norm)
                pred_prob = knn.predict_proba(X_test_norm)
            elif (model_name == "SVM_Classification"):
                svm = OneVsRestClassifier(SVC(kernel='linear', probability=True))
                svm.fit(X_train_norm, y_train)
                y_pred = svm.predict(X_test_norm)
                pred_prob = svm.predict_proba(X_test_norm)
            elif (model_name == "LR_Classification"):
                lr = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
                lr.fit(X_train_norm, y_train)
                y_pred = lr.predict(X_test_norm)
                pred_prob = lr.predict_proba(X_test_norm)
            elif (model_name == "DT_Classification"):
                dt = OneVsRestClassifier(DecisionTreeClassifier())
                dt.fit(X_train_norm, y_train)
                y_pred = dt.predict(X_test_norm)
                pred_prob = dt.predict_proba(X_test_norm)
            else:
                raise ValueError(
                    "Invalid model name specified. Please choose one of 'Random Forest Classification', 'Multilayer Perceptron', 'K-Nearest Neighbors', 'Support Vector Machine', 'Logistic Regression', 'Decision Tree Classification'.")

            y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

            fpr = {}
            tpr = {}
            thresh = {}
            roc_auc = dict()

            n_class = classes.shape[0]

            for i in range(n_class):
                fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

                plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)' % (classes[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'b--')
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.title('Multiclass ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive rate')
            plt.legend(loc='lower right')
            st.pyplot(plt.gcf())
            return
        # if y.nunique() > 2:
        #     label_encoder = LabelEncoder()
        #     label_encoder.fit(y)
        #     y = label_encoder.transform(y)
        #     classes = label_encoder.classes_
        #
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #
        #     min_max_scaler = MinMaxScaler()
        #     X_train_norm = min_max_scaler.fit_transform(X_train)
        #     X_test_norm = min_max_scaler.fit_transform(X_test)
        #
        #     mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
        #     mlp.fit(X_train_norm, y_train)
        #     y_pred = mlp.predict(X_test_norm)
        #     pred_prob = mlp.predict_proba(X_test_norm)
        #
        #     y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        #
        #     fpr = {}
        #     tpr = {}
        #     thresh = {}
        #     roc_auc = dict()
        #
        #     n_class = classes.shape[0]
        #
        #     for i in range(n_class):
        #         fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
        #         roc_auc[i] = auc(fpr[i], tpr[i])
        #
        #         plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)' % (classes[i], roc_auc[i]))
        #
        #     plt.plot([0, 1], [0, 1], 'b--')
        #     plt.xlim([0, 1])
        #     plt.ylim([0, 1.05])
        #     plt.title('Multiclass ROC curve')
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive rate')
        #     plt.legend(loc='lower right')
        #     plt.show()
        #
        #     st.pyplot(plt.gcf())
        #
        #     return
        else:
            y_encoded = le.fit_transform(y)
            y_pred_encoded = le.transform(y_pred)

            fpr, tpr, _ = roc_curve(y_encoded.ravel(), y_pred_encoded.ravel())

            fig, ax = plt.subplots()

            sns.lineplot(x=fpr, y=tpr,ax=ax)

            ax.set_xlabel("False Positive Rate")

            ax.set_ylabel("True Positive Rate")

            ax.set_title("ROC Curve")

            st.pyplot(fig)

    # elif result_opt == "Feature Importances":
    #
    #     if hasattr(models, 'feature_importances_'):
    #
    #         feature_importances = pd.Series(models.feature_importances_, index=X.columns)
    #
    #     elif hasattr(models, 'coef_'):
    #
    #         feature_importances = pd.Series(models.coef_[0], index=X.columns)
    #
    #     else:
    #
    #         st.warning("Model doesn't support feature importances")
    #
    #         return
    #
    #     feature_importances = feature_importances.sort_values(ascending=False)
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #
    #     sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax)
    #
    #     ax.set_xlabel("Feature Importance Score")
    #
    #     ax.set_ylabel("Features")
    #
    #     ax.set_title("Feature Importances")
    #
    #     st.pyplot(fig)
def actvspred(y,y_pred,graph_header):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.lineplot(x=range(len(y)), y=y, label="Actual")
    ax = sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title(graph_header)
    st.pyplot(fig)
    plt.close((fig))

# def show_multiclass(y, y_pred):
#
#     # Binarize the y and y_pred labels
#     y_binarized = label_binarize(y, classes=np.unique(y))
#     y_pred_binarized = label_binarize(y_pred, classes=np.unique(y))
#
#     # Compute the false positive rate, true positive rate, and area under the curve for each class
#     n_classes = y_binarized.shape[1]
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred_binarized[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and area under the curve
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_pred_binarized.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # Plot the ROC curve for each class and micro-average ROC curve
#     fig, ax = plt.subplots()
#     for i in range(n_classes):
#         ax.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#
#     ax.plot([0, 1], [0, 1], 'k--', label='random guess')
#     ax.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
#
#     # Set the x and y limits and labels, and add a legend
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlabel('False Positive Rate')
#     ax.set_ylabel('True Positive Rate')
#     ax.set_title('Multiclass ROC Curve')
#     ax.legend(loc="lower right")
#     plt.show()
