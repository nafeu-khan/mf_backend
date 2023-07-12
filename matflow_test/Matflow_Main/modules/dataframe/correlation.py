# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from ...modules import utils

# def correlation(data):
# 	st.title("Feature Correlation")
#
# 	num_var = utils.get_numerical(data)
# 	col1, col2 = st.columns([8,2])
#
# 	col2.markdown("#")
# 	select_all = col2.checkbox("Select all", True, key="correlation_select_all")
#
# 	if select_all:
# 		correlation_var = col1.multiselect(
# 				"Columns",
# 				num_var,
# 				num_var,
# 				key="correlation_var"
# 			)
# 	else:
# 		correlation_var = col1.multiselect(
# 				"Columns",
# 				num_var,
# 				key="correlation_var"
# 			)
#
# 	col1, col2, col3 = st.columns([4,4,2.02])
# 	correlation_method = col1.selectbox(
# 			"Method",
# 			["pearson", "kendall", "spearman"],
# 			key="correlation_method"
# 		)
#
# 	display_type = col2.selectbox(
# 			"Display Type",
# 			["Table", "Heatmap", "Feature Pair"],
# 			key="correlation_display_type"
# 		)
#
# 	if correlation_var:
# 		if display_type == "Table":
# 			col3.markdown("#")
# 			bg_gradient = col3.checkbox("Gradient", key="correlation_bg_gradient")
# 		elif display_type == "Heatmap":
# 			col3.markdown("#")
# 			annot = col3.checkbox("Annotate", key="correlation_annot")
# 		else:
# 			col3.markdown("#")
# 			bg_gradient = col3.checkbox("Gradient", key="correlation_bg_gradient")
#
#
#
# 		correlation_data = data[correlation_var].corr(correlation_method)
# 		if display_type == "Table":
# 			display_table(correlation_data, bg_gradient)
# 		elif display_type == "Heatmap":
# 			display_heatmap(correlation_data, annot)
# 		else:
# 			display_pair(correlation_data, bg_gradient)
#
# def display_table(correlation_data, bg_gradient):
# 	if bg_gradient:
# 		st.dataframe(correlation_data.style.background_gradient())
# 	else:
# 		st.dataframe(correlation_data)
import plotly.graph_objects as go
import plotly.io as pio
from django.http import JsonResponse

# def display_heatmap(correlation_data):
#     decimal = float(0)
#     annot=True
#     if annot:
#         decimal = 3  # Default decimal value
#         # You can add a user input for the decimal value if desired
#
#     fig = go.Figure(data=go.Heatmap(
#         z=correlation_data.round(2),
#         x=correlation_data.columns,
#         y=correlation_data.columns,
#         colorscale='Viridis',
#         colorbar=dict(title='Correlation'),
#         hovertemplate='Value: %{z:.2f}<extra></extra>'  # Format for the hover text
#     ))
#
#     fig.update_layout(
#         title='Feature Correlation Heatmap',
#         xaxis=dict(title='Features'),
#         yaxis=dict(title='Features'),
#         hovermode='closest'  # Set hovermode to 'closest' to show the closest data point value
#     )
#     # Add annotations if desired
#     if annot:
#         for i in range(len(correlation_data.columns)):
#             for j in range(len(correlation_data.columns)):
#                 x = correlation_data.columns[j],
#                 y = correlation_data.columns[i],
#                 if isinstance(x,str) or isinstance(y,str):
#                     fig.add_annotation(
#                         x=correlation_data.columns[j],
#                         y=correlation_data.columns[i],
#                         text=f'{float(correlation_data.iloc[i, j]):.{int(decimal)}f}',
#                         showarrow=False,
#                         font=dict(color='white' if float(correlation_data.iloc[i, j]) > 0.5 else 'black')
#                     )
#                 else:
#                     continue
#
#     fig_json = pio.to_json(fig)
#     return JsonResponse({'graph': fig_json})



import numpy as np

def display_heatmap(correlation_data):
    decimal = float(0)
    correlation_data = correlation_data.set_index(correlation_data.columns[0])
    correlation_data.index = correlation_data.index.set_names(None)
    annot=True
    if annot:
        decimal = 3  # Default decimal value
        # You can add a user input for the decimal value if desired

    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.round(2),
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
        hovertemplate='Value: %{z:.2f}<extra></extra>'  # Format for the hover text
    ))

    fig.update_layout(
        title='Feature Correlation Heatmap',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features'),
        hovermode='closest'  # Set hovermode to 'closest' to show the closest data point value
    )
    # Add annotations if desired
    if annot:
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                value = correlation_data.iloc[i, j]
                if isinstance(value, (int, float, np.number)):
                    x = correlation_data.columns[j],
                    y = correlation_data.columns[i],
                    fig.add_annotation(
                        x=correlation_data.columns[j],
                        y=correlation_data.columns[i],
                        text=f'{float(value):.{int(decimal)}f}',
                        showarrow=False,
                        font=dict(color='white' if float(value) > 0.5 else 'black')
                    )

    fig_json = pio.to_json(fig)
    return JsonResponse( fig_json)




#
# def display_pair(correlation_data, bg_gradient,feature1,feature2,higher_than,drop_perfect,convert_abs):
# 	features = correlation_data.columns.to_list()
# 	features.insert(0, "-")
#
# 	col1, col2, _ = st.columns([2.5,2.5,5])
# 	drop_perfect = col1.checkbox("Drop Perfect", key="correlation_drop_perfect")
# 	convert_abs = col2.checkbox("Absolute Value", key="convert_absolute")
#
# 	if convert_abs:
# 		# convert to absolute value to take negative correlation into consideration and then sort by the highest correlation
# 		sorted_corr = correlation_data \
# 						.abs() \
# 						.unstack() \
# 						.sort_values(ascending=False) \
# 						.reset_index() \
#
# 	else:
# 		sorted_corr = correlation_data \
# 						.unstack() \
# 						.sort_values(ascending=False) \
# 						.reset_index() \
#
# 	sorted_corr.rename(
# 			columns = {
# 				"level_0": "Feature 1",
# 				"level_1": "Feature 2",
# 				0: 'Correlation Coefficient'
# 			}, inplace=True
# 		)
#
# 	if drop_perfect:
# 		sorted_corr = sorted_corr.drop(sorted_corr[sorted_corr['Correlation Coefficient'] == 1.0].index)
#
# 	if higher_than:
# 		sorted_corr = sorted_corr[sorted_corr['Correlation Coefficient'] > higher_than].reset_index(drop=True)
#
# 	if feature1 != "-" and feature2 == "-":
# 		sorted_corr = sorted_corr.loc[sorted_corr["Feature 1"] == feature1].reset_index(drop=True)
# 	elif feature1 == "-" and feature2 != "-":
# 		sorted_corr = sorted_corr.loc[sorted_corr["Feature 2"] == feature2].reset_index(drop=True)
# 	elif feature1 != "-" and feature2 != "-":
# 		if feature1 == feature2:
# 			# drop observation with same features but different column
# 			sorted_corr.drop(sorted_corr.iloc[1::2].index, inplace=True)
# 			sorted_corr = sorted_corr.loc[(sorted_corr["Feature 1"] == feature1) | (sorted_corr["Feature 2"] == feature2)].reset_index(drop=True)
# 		else:
# 			sorted_corr = sorted_corr.loc[(sorted_corr["Feature 1"] == feature1) | (sorted_corr["Feature 2"] == feature2)].reset_index(drop=True)
#
# 	else:
# 		sorted_corr = sorted_corr.drop(sorted_corr.iloc[1::2].index).reset_index(drop=True)
#
# 	return sorted_corr
