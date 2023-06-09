import streamlit as st
import numpy as np

from modules import utils
from modules.classes import dropper

def dropping(data, data_opt):
	temp_name = ''
	variables = utils.get_variables(data)
	cat_var = utils.get_categorical(data)
	null_var = utils.get_null(data)
	blank_var=utils.get_blank_column(data)
	option_dict = {
			"All": variables,
			"Categorical": cat_var, 
			"With Null": null_var, 
			"Blank": blank_var
		}

	option = st.radio(
			"Default Columns",
			option_dict.keys(),
			index=3,
			key="drop_default_options",
			horizontal=True
		)

	col1, col2 = st.columns([7.5, 2.5])
	drop_var = col1.multiselect(
			"Select Columns",
			variables,
			option_dict[option],
			key="drop_var"
		)

	col2.markdown("#")
	add_pipeline = col2.checkbox("Add To Pipeline", True, key="drop_add_pipeline")
	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True,key='drop_col')

	if save_as:
		temp_name=col2.text_input('New Dataset Name')
	if st.button("Submit", key="drop_submit"):

		if drop_var:
			drp = dropper.Dropper(drop_var)
			new_value = drp.fit_transform(data)

			if add_pipeline:
				name = f"Drop {', '.join(drop_var)} column"
				utils.add_pipeline(name, drp)
			if utils.update_value(data_opt, new_value,temp_name,save_as):
				st.success("Success")

			utils.rerun()

		else:
			st.warning("Select columns to drop")

def drop_raw(data,data_opt):
	temp_name = ''
	variables = utils.get_variables(data)
	cat_var = utils.get_categorical(data)
	null_var = utils.get_null(data)

	option_ = [
		"With Null"
	]

	option = st.radio(
		"Default Columns",
		option_,
		key="drop_default_row",
		horizontal=True
	)

	col1, col2 = st.columns([7.5, 2.5])

	add_pipeline = col2.checkbox("Add To Pipeline", True, key="drop_add_row_pipeline")

	drop_var = col1.multiselect(
		"Select Columns",
		variables,
		null_var,
		key="drop_var1"
	)
	# remove all rows with null values
	# data = data.dropna()
	#
	# # remove all rows with blank values
	# data = data.replace('', np.nan).dropna()
	#
	# remove all rows with null or blank values in specific columns

	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True)

	if save_as:
		temp_name = col2.text_input('New Dataset Name',key='drop_row1')
	if st.button("Submit", key="drop_submit_row"):
		if drop_var:
			print(drop_var)
			new_value = data.dropna(subset=drop_var)
			if add_pipeline:
				name = f"Drop {', '.join(drop_var)} column"
				utils.add_pipeline(name, new_value)
			utils.update_value(data_opt, new_value,temp_name,save_as)
			st.success("Success")

			utils.rerun()
		else:
			st.warning("Select columns to drop")

	#
	# # remove all rows with null or blank values in categorical columns
	# data = data.dropna(subset=['categorical_column']).replace('', np.nan).dropna(subset=['another_categorical_column'])
	#
	# # remove all rows with null values in specific columns and fill null values with mean or median
	# data['column1'].fillna(data['column1'].mean(), inplace=True)
	# data['column2'].fillna(data['column2'].median(), inplace=True)
