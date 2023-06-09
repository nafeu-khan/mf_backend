import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def model_report(split_dataset_name, models):
    result_df = pd.DataFrame()
    for i in st.session_state.has_models[split_dataset_name]:
        result_df = pd.concat([result_df, models.get_result(i)], ignore_index=True)

    col1, col2 = st.columns(2)
    display_type = col1.selectbox(
        "Display Type",
        ["Table", "Graph"]
    )

    if display_type == "Table":
        col2.markdown("#")
        include_data = col2.checkbox("Include Data")
        report_table(result_df, include_data)
    else:
        report_graph(result_df, col2)


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

def report_graph(data, col):
    model_data = data.drop(columns=['Train Data', 'Test Data', 'Model Name'])
    cmap = plt.cm.get_cmap('Set3', len(model_data))
    result_df = model_data
    column=pd.DataFrame()
    orientation = st.selectbox('Select Orientation', ['Vertical', 'Horizontal'])
    for i in range(2):
        st.write('#')
    display_result = st.radio(
        "Display Result",
        ["All", "Train", "Test", "Custom"],
        index=2,
        horizontal=True
    )
    if display_result == "All":
        column = model_data

    elif display_result == "Train":
        colms = result_df.columns[result_df.columns.str.contains("Train")].to_list()
        column = model_data[colms]

    elif display_result == "Test":
        colms = result_df.columns[result_df.columns.str.contains("Test")].to_list()
        column = model_data[colms]

    elif display_result == "Custom":
        try:
            selected_columns = st.multiselect(
                "Columns",
                model_data.columns,
                []
            )
            if len(selected_columns) > 0:
                column = model_data[selected_columns]
            else :
                st.warning("Please select at least one column")
        except ValueError:
            st.warning("Please select at least one column")

    if orientation == 'Vertical':
        fig, ax = plt.subplots(nrows=1, ncols=len(column.columns), figsize=(16,8))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i, col in enumerate(column.columns):
            for j, row in enumerate(column.iterrows()):
                # Create the bar plot for the current column
                ax[i].bar(j, row[1][i], color=cmap(j), label=list(data['Model Name'].values)[j])
                ax[i].set_xticklabels([])
                ax[i].set_xlabel(col)
        ax[0].set_ylabel("Value")
        ax[-1].legend(loc='upper left', bbox_to_anchor=(0, 1.3))
        fig.subplots_adjust(top=0.85 + 0.05 * len(data))

    elif orientation == 'Horizontal':
        fig, ax = plt.subplots(nrows=len(model_data.columns), ncols=1, figsize=(10,8+len(model_data.columns)))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        for i, col in enumerate(model_data.columns):
            for j, row in enumerate(model_data.iterrows()):
                # Create the bar plot for the current column
                ax[i].barh(j, row[1][i], color=cmap(j), label=list(data['Model Name'].values)[j])
                ax[i].set_yticklabels([])
                ax[i].set_ylabel(col)
        ax[-1].set_xlabel("Value")
        ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.5))
        fig.subplots_adjust(top=0.85 + 0.05 * len(data))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
