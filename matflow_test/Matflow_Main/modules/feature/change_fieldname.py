import streamlit as st
from modules import utils

def change_field_name(file_,opt):
    temp_name=''
    fixed_li = file_.columns.values.tolist()
    col1,col2,col3=st.columns(3)
    with col1:
        n_iter = st.number_input(
            "Number of Columns",
            1, len(fixed_li), 1,
            key="change_fname_n_rows"
        )
    col1, col2 = st.columns([5, 5])
    selected = []

    for i in range(n_iter):
        with col1:
            var = st.selectbox(
                "Field Name",
                options=fixed_li,
                key=f"change_fname_var_{i}"
            )
            selected.append(var)
            fixed_li = [key for key in fixed_li if key not in selected]
        with col2:
            var2 = st.text_input('New Field Name',
                key=f"change_fname_rename_{i}"
            )
    col1, col2, c0 = st.columns([2, 2, 2])
    save_as = col1.checkbox('Save as New Dataset', True, key="change_field_name_check")

    if save_as:
        temp_name = col2.text_input('New Dataset Name', key="change_field_name_name")

    if st.button('save'):
        change=True
        temp=file_
        for i in range(n_iter):
            for j in range(n_iter):
                if st.session_state['change_fname_var_'+str(i)]==st.session_state['change_fname_rename_'+str(i)] or len(st.session_state['change_fname_rename_'+str(i)])==0:
                    change=False
        if not change:
            st.warning('Field name can\'t be same or empty')
        else:
            for i in range(n_iter):
                    temp=temp.rename(columns={st.session_state['change_fname_var_'+str(i)]:st.session_state['change_fname_rename_'+str(i)]})
            utils.update_value(opt,temp,temp_name,save_as)
            st.session_state.btn_state = False
            st._rerun()

