import base64
import io
import json

import pandas
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression, \
    SelectFromModel
from ...subpage import customFeatureSelection

def visualize(X, y, selected_features_df):
    response_data = {}

    if not y.dtype == 'object':
        if len(selected_features_df) >= 2:
            # Get the two best features
            feature1 = selected_features_df.iloc[0]['Feature']
            feature2 = selected_features_df.iloc[1]['Feature']

            # Create a scatter plot with hue as the target variable
            fig, ax = plt.subplots()
            sns.scatterplot(x=X[feature1], y=X[feature2], hue=y, ax=ax)
            ax.set_title(f"Scatter Plot of {feature1} vs. {feature2}")
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)

            # Convert the figure to JSON-serializable data
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            response_data['scatter_plot'] = base64.b64encode(buffer.read()).decode("utf-8")

    # Create a bar plot of the selected features and their scores
    fig, ax = plt.subplots()
    ax.bar(selected_features_df['Feature'], selected_features_df['Score'])
    ax.set_xticklabels(selected_features_df['Feature'], rotation=90)
    ax.set_title("Selected Features and Scores")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Score")

    # Convert the figure to JSON-serializable data
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    response_data['bar_plot'] = base64.b64encode(buffer.read()).decode("utf-8")

    return response_data

def feature_selection(file,data,table_name, target_var, method, score_func, show_graph,best_Kfeature):
    response_data = {}

    # Separate target variable and features
    X = data.drop(columns=target_var)
    y = data[target_var]

    # Auto-select task based on target variable
    if y.dtype == 'object':
        task = 'classification'
    else:
        task = 'regression'

    # Select feature selection method and score function based on task
    if task == 'classification':
        if method == 'SelectKBest':
            selector = SelectKBest(score_func=eval(score_func), k=best_Kfeature)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=0)
            selector = SelectFromModel(estimator=estimator)
    else:
        if method == 'SelectKBest':
            selector = SelectKBest(score_func=eval(score_func), k=best_Kfeature)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=0)
            selector = SelectFromModel(estimator=estimator)

    # Perform feature selection
    try:
        selector.fit(X, y)
    except ValueError:
        response_data["error"] = "Feature selection failed. Please check if the selected score function is compatible with the data."
        return response_data

    selected_features = X.columns[selector.get_support()]

    # Display selected features and scores
    if method == 'SelectKBest':
        selected_scores = selector.scores_[selector.get_support()]
    else:
        selected_scores = selector.estimator_.feature_importances_[selector.get_support()]
    selected_features_df = pd.DataFrame({
        'Feature': selected_features,
        'Score': selected_scores
    })
    selected_features_df.sort_values('Score', ascending=False, inplace=True)
    selected_features_df.reset_index(drop=True, inplace=True)

    # Prepare response data
    response_data["selected_features"] = selected_features_df.to_dict(orient='records')

    # Call customFeatureSelection if needed
    if method == 'Best Overall Features':
        #need to catch kfold and display_opt
        kfold=file.get('k_fold')
        display_opt=file.get("display_opt")
        custom_feature_data = customFeatureSelection.feature_selection(data, table_name, target_var, task, kfold, display_opt, selected_features=None)
        response_data["custom_feature_data"] = custom_feature_data

    if show_graph:
        graph_data = visualize(X, y, selected_features_df)
        response_data["graph_data"] = graph_data

    # Return selected features and scores as JSON-serializable response data
    return json.dumps(response_data, indent=4)