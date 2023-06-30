import pandas as pd
from django.http import JsonResponse


def append (file):
    # append_name = file.get('select_dataset_you_wanna_append')
    # file_name = file.get('new_dataset_name')
    data=pd.DataFrame(file.get("file"))
    tmp = pd.DataFrame(file.get("file2"))

    temp2 = tmp.append(data)
    # temp2 = temp2.reset_index()
    print(f"type temp2 = {type (temp2)} tmp = {type(tmp) }data = {type(data)}")

    new_value =temp2 .to_dict(orient="records")
    print(f"type new = {type (new_value)}")

    return JsonResponse(new_value, safe=False)
  #
  # #
  # li = []
  #           for i in st.session_state.dataset.data.keys():
  #               if i != table_name:
  #                   li.append(i)
  #
  #           # append_name = st.selectbox('Select Dataset You Wanna Append', li)
  #           # file_name = st.text_input('New Dataset Name', autocomplete='off', key='append')
  #           # st.write('#')
  #           # if st.button('Append', type='primary'):
  #           #     if len(file_name) == 0:
  #           #         st.error('Name can\'t be empty')
  #               else:
  #                   tmp = pd.DataFrame(data)
  #                   try:
  #                       temp2 = tmp.append(dataset[append_name])
  #                       temp2 = temp2.reset_index()
  #                       # st.session_state.dataset.add(file_name, temp2)
  #                       # st._rerun()
  #                   # except Exception as e:
  #                   #     st.warning(e)