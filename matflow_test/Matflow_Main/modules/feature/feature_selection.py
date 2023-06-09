import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression, \
    SelectFromModel
from subpage import Feature_Selection
from modules import utils

def visualize(X, y, selected_features_df):
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
            st.pyplot(fig)

    # Create a bar plot of the selected features and their scores
    fig, ax = plt.subplots()
    ax.bar(selected_features_df['Feature'], selected_features_df['Score'])
    ax.set_xticklabels(selected_features_df['Feature'], rotation=90)
    ax.set_title("Selected Features and Scores")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Score")
    st.pyplot(fig)


def feature_selection(dataset, table_name):
    try:
        data = dataset
    except ValueError:
        return
    c0,col1,col2,c1=st.columns([0.1,2,2,0.1])
    _c0,_col1,_col2,_c1=st.columns([0.1,3,2,0.1])

    with col1:
        target_var = st.selectbox(
            "Target Variable",
            utils.get_variables(data),
            index=len(data.columns)-1,
            key="model_target_var"
        )
    # Separate target variable and features
    X = data.drop(columns=target_var)
    y = data[target_var]

    # Auto-select task based on target variable
    if y.dtype == 'object':
        task = 'classification'
    else:
        task = 'regression'
    with col2:
        try:
            method = st.selectbox('Select feature selection method:', ['Best Overall Features','SelectKBest', 'Mutual Information'], key='feature_selection_method')
        except ValueError:
            st.error("Invalid value for feature selection method.")
            return


    # Select feature selection method and score function based on task
    if task == 'classification':

        if method=='Best Overall Features':
            Feature_Selection.feature_selection(st.session_state.dataset.data,table_name,target_var, 'classification')
            # if not feature_select_data.empty:
            #     Feature_Selection.feature_graph(feature_select_data,'classification')
            return
        elif method == 'SelectKBest':
            try:
                # Select number of features to keep
                with _col1:
                    k = st.slider('Select number of features to keep:', min_value=1, max_value=len(X.columns), value=1,
                                  step=1)
                with _col2:
                    score_func = st.selectbox('Select score function:', ['f_classif', 'mutual_info_classif'], key='selectkbest_score_func')
            except ValueError:
                st.error("Invalid value for score function.")
                return
        else:
            score_func = 'mutual_info_classif'
    else:

        if method=='Best Overall Features':
            Feature_Selection.feature_selection(st.session_state.dataset.data,table_name,target_var, 'regression')
            # if not feature_select_data.empty:
            #     Feature_Selection.feature_graph(feature_select_data,'regression')
            return
        elif method == 'SelectKBest':
            try:
                with _col1:
                    k = st.slider('Select number of features to keep:', min_value=1, max_value=len(X.columns), value=1,
                                  step=1)
                with _col2:
                    score_func = st.selectbox('Select score function:', ['f_regression', 'mutual_info_regression'], key='selectkbest_score_func')
            except ValueError:
                st.error("Invalid value for score function.")
                return
        else:
            score_func = 'mutual_info_regression'

    # Perform feature selection
    try:
        if method == 'SelectKBest':
            selector = SelectKBest(score_func=eval(score_func), k=k)
        else:
            if task == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=0)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=0)
            selector = SelectFromModel(estimator=estimator)
        selector.fit(X, y)
    except ValueError:
        st.error("Feature selection failed. Please check if the selected score function is compatible with the data.")
        return

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
    selected_features_df.sort_values('Score',ascending=False,inplace=True)
    selected_features_df.reset_index(drop=True, inplace=True)
    with _col2:
        st.write('#')
        st.write('#')
        show_graph=st.checkbox('Show Graph')

    c0,col1,c1=st.columns([0.1,4,0.1])
    with col1:
        st.write('Selected Features and Scores:')

        st.dataframe(selected_features_df)

        if show_graph:
            visualize(X,y,selected_features_df)

    # Return selected features and scores as dataframe

    X_selected = selector.transform(X)
    return pd.DataFrame(X_selected, columns=selected_features)
