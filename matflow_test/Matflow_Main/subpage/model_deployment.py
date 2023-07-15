import pandas as pd
import streamlit as st


def model_deployment(file):
    model_name = file.get("model_name")
    dataset = pd.DataFrame(file.get("model_name"))

    model = file.get("models")
    train_data_name = pd.DataFrame(file.get('train_name'))
    test_data_name=dataset['test_name']
    train_data = pd.DataFrame(file.get('train'))
    test_data = pd.DataFrame(file.get('test'))
    target_var = dataset['target_var']


    col_names_all = []
    for i in train_data.columns:
        if i == target_var:
            continue
        col_names_all.append(i)
    rad=1
    if rad == 'All Columns':
        col_names = col_names_all
    else:
        col_names = st.multiselect('Custom Columns', col_names_all, help='Other values will be 0 as default value')
    col_names= file.get("col_names")
    col1, col2, col3, col4 = st.columns([4, 1, 2, 0.5])
    prediction = ['']
    correlations = train_data[col_names + [target_var]].corr()[target_var]

    with col1:
        st.header('INPUT')
        for i in col_names:
            threshold=train_data[i].abs().max()
            arrow,threshold = ('**:green[↑]**',threshold) if correlations[i] >= 0 else ('**:red[↓]**',-threshold)
            space='&nbsp;'*150
            st.number_input(i + space + str(threshold)+' ' + arrow,value=threshold, key=i)

        st.write('#')
        if st.button('Submit', type="primary"):
            X = [st.session_state[i] if i in col_names else 0 for i in col_names_all]

            prediction = model.get_prediction(model_name, [X])

            def get_prediction(self, model_name, X):
                return self.model[model_name].predict(X)

    # with col2:
    #     st.write('#')
    #     for i in col_names:
    #         threshold = train_data[i].abs().max()
    #         arrow, threshold = ('**:green[↑]**', threshold) if correlations[i] >= 0 else ('**:red[↓]**', -threshold)
    #         st.write(str(threshold)+' '+arrow)
    #         st.write('#')


    with col3:
        st.header('PREDICTION')
        st.header('#')
        st.write('#')
        st.write('#')
        st.write('#')
        st.markdown(f'<h4> {target_var.upper()} : {prediction[0]}</h4>',
                    unsafe_allow_html=True
                    )

