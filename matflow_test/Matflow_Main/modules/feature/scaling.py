import streamlit as st
from django.http import JsonResponse

from ...modules import utils
from ...modules.classes import scaler

def scaling(file, data_opt):
	data=file.get("data")
	variables = utils.get_variables(data)
	num_var = utils.get_numerical(data)
	cat_var = utils.get_categorical(data)

	default_dict = {
			"Blank": [],
			"All": variables,
			"Numerical": num_var,
			"Categorical": cat_var,
		}

	col1, col2, col3 = st.columns([4,4,2.5])
	col_options = file.get("options")

	method = col2.selectbox("method")

	add_pipeline = file.get("add_to_pipeline")

	default = file.get("default_value")

	columns =file.get("col_options")

	if st.button("Submit", key="scaling_submit"):
		if col_options == "Select All Except":
			columns = [var for var in variables if var not in columns]

		sc = scaler.Scaler(method, columns)
		new_value = sc.fit_transform(data)

		new_value = new_value.to_dict(orient="records")
		return JsonResponse(new_value, safe=False)
