import io
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from django.http import HttpResponse
from django.http import JsonResponse
import base64

def model_report(file):
    result_df = pd.DataFrame(file.get("file"))
    # display_type =file.get( "Display Type")
    # if display_type == "Table":
    #     include_data = file.get("Include Data")
    #     report_table(result_df, include_data)
    # else:
    return  report_graph(result_df,file)
def report_table(result_df, include_data):
    cols = result_df.columns
    if not include_data:
        cols = [col for col in cols if col not in ["Train Data", "Test Data"]]

    display_result = st.radio(
        "Display Result",
        ["All", "Train", "Test", "Custom"],
        index=2,
        horizontal=True
    )

    if display_result == "Train":
        cols = result_df.columns[result_df.columns.str.contains("Train")].to_list()
        if include_data:
            cols.insert(0, "Model Name")
        else:
            cols[0] = "Model Name"

    elif display_result == "Test":
        cols = result_df.columns[result_df.columns.str.contains("Test")].to_list()
        if include_data:
            cols.insert(0, "Model Name")
        else:
            cols[0] = "Model Name"

    elif display_result == "Custom":
        cols = st.multiselect(
            "Columns",
            cols,
            ["Model Name"]
        )
    st.dataframe(result_df[cols])

def report_graph(data, file):
    model_data=data
    try:
        model_data = data.drop(columns=['Train Data', 'Test Data', 'Model Name'])
    except:
        model_data=data
        pass
    cmap = plt.cm.get_cmap('Set3', len(model_data))
    result_df = model_data
    column=pd.DataFrame()
    orientation = file.get("Select Orientation")
    display_result = file.get("Display Result")
    if display_result == "All":
        column = model_data

    elif display_result == "Train":
        colms = result_df.columns[result_df.columns.str.contains("Train")].to_list()
        column = model_data[colms]

    elif display_result == "Test":
        colms = result_df.columns[result_df.columns.str.contains("Test")].to_list()
        column = model_data[colms]

    elif display_result == "Custom":
        selected_columns =file.get("Columns")
        if len(selected_columns) > 0:
            column = model_data[selected_columns]

    fig, ax = plt.subplots(nrows=1, ncols=len(column.columns), figsize=(16,8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    if orientation == 'Vertical':

        for i, col in enumerate(column.columns):
            for j, row in enumerate(column.iterrows()):
                # Create the bar plot for the current column
                ax[i].bar(j, row[1][i], color=cmap(j), label=list(data['name'].values)[j])
                ax[i].set_xticklabels([])
                ax[i].set_xlabel(col)
        ax[0].set_ylabel("Value")
        ax[-1].legend(loc='upper left', bbox_to_anchor=(0, 1.3))
        fig.subplots_adjust(top=0.85 + 0.05 * len(data))

    elif orientation == 'Horizontal':
        for i, col in enumerate(model_data.columns):
            for j, row in enumerate(model_data.iterrows()):
                # Create the bar plot for the current column
                ax[i].barh(j, row[1][i], color=cmap(j), label=list(data['name'].values)[j])
                ax[i].set_yticklabels([])
                ax[i].set_ylabel(col)
        ax[-1].set_xlabel("Value")
        ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.5))
        fig.subplots_adjust(top=0.85 + 0.05 * len(data))

    plt.tight_layout()
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    plt.close(fig)
    image_stream.seek(0)

    # Encode the image stream as base64
    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    # Create the Plotly graph with the base64-encoded image and increase size
    graph = go.Figure(go.Image(source=f'data:image/png;base64,{image_base64}'))
    graph.update_layout(font=dict(family="Arial", size=12), width=1000, height=800,
                        # xaxis=dict(editable=True),yaxis=dict(editable=True)
                        )
    # Convert the graph to HTML and send as a response
    html_content = pio.to_html(graph, full_html=False)
    response = HttpResponse(content_type='text/html')
    response.write(html_content)
    print(5)
    # Return the graph JSON data
    graph_json = graph.to_json()
    print(graph_json)
    return JsonResponse(graph_json, safe=False)
