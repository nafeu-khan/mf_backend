import streamlit as st
import os
from modules.dataset import read
from modules import utils
from sklearn.model_selection import train_test_split

def split_dataset(dataset,data_opt):
    list_data = dataset.list_name()
    ds_name= os.path.splitext(data_opt)[0]
    if list_data:
        col1,col2=st.columns(2)

        # if  read.validate(split_dataset_root_name, list_data):

        data = dataset.get_data(data_opt)
        with col1:
            target_var = col1.selectbox(
                f":blue[Target Variable]",
                utils.get_variables(data),
                key="model_target_var"
            )

            if data[target_var].dtype == "float64" or data[target_var].dtype == "int64":
                type = 'Regressor'
                st.write('#')
                st.write(f" ##### {target_var} _is **:blue[Continuous]**._")

            else:
                type='Classification'
                st.write('#')
                st.write(f" ##### {target_var} _is **:red[Categorical]**._")

        split_dataset_name = f"{ds_name}_{target_var}"
        with col2:
            variables = utils.get_variables(data, add_hypen=True)
            stratify = col2.selectbox(
                f":violet[Stratify]",
                variables,
                key="split_stratify"
            )


        col1, col2,col4 = st.columns([3, 4,1])

        test_size = col1.number_input(
            f":green[Test Size]",
            0.0, 1.0, 0.5,
            key="split_test_size"
        )

        random_state = col2.slider(
            f":orange[Random State]",
            0, 1000, 1,
            key="split_random_state"
        )
        col1, col2 = st.columns(2)
        shuffle = col4.checkbox("Shuffle", True, key="split_shuffle")
        if(shuffle):
            s="suffled"
        else:
            s=""
        if stratify=="-":
            stfy= ""
        else :
            stfy=f"-stratify= {stratify}"
        # split_dataset_name = f"{split_dataset_name}{stfy}-[{test_size}]-[{random_state}]-{s}"
        train_name = col1.text_input(
            "Train Data Name",
            f"train_{split_dataset_name}",
            key="train_data_name"
        )
        test_name = col2.text_input(
            "Test Data Name",
            f"test_{split_dataset_name}",
            key="test_data_name"
        )

        col4.markdown("#")
        col1,col2=st.columns(2)
        split_dataset_name=col1.text_input(
            "**Splitted Dataset Name**",
           split_dataset_name ,
            key="split_name"
        )
        # f"{ds_name}-:blue[{target_var}]-:violet[{stfy}] - :green[{test_size}] - :orange[{random_state}] -{s}"

        if st.button("Submit", key="split_submit_button"):

            if 'splitted_data' not in st.session_state:
                st.session_state.splitted_data = {}

            split_list = st.session_state.splitted_data

            is_valid = [read.validate(name, list_data) for name in [train_name, test_name]]

            if all(is_valid):
                stratify = None if (stratify == "-") else stratify
                X = data
                y = data[target_var]
                X_train, X_test = train_test_split(X, test_size=test_size,random_state=random_state)
                split_list.update({split_dataset_name:{'train_name': train_name, 'test_name': test_name,
                                                  'target_var':target_var,'type':type}})
                dataset.add(train_name, X_train)
                dataset.add(test_name, X_test)
                st.success("Success")
                utils.rerun()

    else:
        st.header("No Dataset Found!")
