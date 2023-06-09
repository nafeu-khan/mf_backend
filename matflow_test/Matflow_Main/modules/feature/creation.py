import streamlit as st
import pandas as pd 
import numpy as np
from modules import utils
from modules.classes import creator

def creation(data, data_opt):
	temp_name=''
	variables = utils.get_variables(data)

	col1, col2, col3, col4 = st.columns([1.6, 3, 2.4, 2.4])
	add_or_mod = col1.selectbox(
			"Options",
			["Add", "Modify"],
			key="add_or_modify"
		)
	st.session_state.add=add_or_mod=='Add'

	method_name = ["New Column","Math Operation", "Extract Text", "Group Categorical", "Group Numerical"]
	if add_or_mod == "Modify":
		method_name.pop(0)
		method_name.append("Replace Values")
		method_name.append("Progress Apply")
	method = col3.selectbox(
		"Method", method_name,
		key="creation_method"
	)


	if add_or_mod == "Add":
		var = col2.text_input(
				"New column name",
				key="add_column_name"
			)
	else:
		var = col2.selectbox(
				"Select column",
				variables,
				key="modify_column_name"
			)


	col4.markdown("#")
	add_pipeline = col4.checkbox("Add To Pipeline", True, key="creation_add_pipeline")


	var = var.strip() # remove whitespace
	if method == "Math Operation":
		math_operation(data, data_opt, var, add_pipeline, add_or_mod)
	elif method == "Extract Text":
		extract_text(data, data_opt, var, add_pipeline, add_or_mod)
	elif method == "Group Categorical":
		group_categorical(data, data_opt, var, add_pipeline, add_or_mod)
	elif method == "Group Numerical":
		group_numerical(data, data_opt, var, add_pipeline, add_or_mod)
	elif add_or_mod=="Modify" and method=="Replace Values":
		replace_values(data,data_opt,var,add_pipeline)
	elif method=='Progress Apply':
		my_progress_apply(data,data_opt,var,add_pipeline)
	elif method=="New Column":
		add_new(data,data_opt,var,add_pipeline)


def math_operation(data, data_opt, var, add_pipeline, add_or_mod):
	temp_name=''
	col1, col2 = st.columns([7,3])
	operation = col1.text_area(
			"New Value Operation",
			key="new_value"
		)
	col1.caption("<math expression> <column name>. example: 10 ** Height " )

	col1.caption(
			"Separate all expression with space (including parenthesis).<br>Example: Weight / ( Height ** 2 )", 
			unsafe_allow_html=True
		)

	col2.markdown("##")
	if col2.button("Show Sample", key="creation_show_sample") and operation:
		crt = creator.Creator("Math Operation", var, operation_string=operation)
		new_value = crt.fit_transform(data)
		st.dataframe(new_value.head())

	col1, col2,c0 = st.columns([2,2,2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name',key="temp_name")

	if st.button("Submit", key="math_submit"):
		if var:
			crt = creator.Creator("Math Operation", column=var, operation_string=operation)
			new_value = crt.fit_transform(data)

			if add_pipeline:
				name = f"{add_or_mod} column {var}"
				utils.add_pipeline(name, crt)

			utils.update_value(data_opt, new_value,temp_name,save_as)
			st.success("Success")

			utils.rerun()
		
		else:
			st.warning("New column name cannot be empty!")


def extract_text(data, data_opt, var, add_pipeline, add_or_mod):
	cat_var = utils.get_categorical(data)

	col1, col2 = st.columns([7,3])
	regex = col1.text_area(
			"Regex pattern", 
			key="extract_regex",
			help=r"Example: ([A-Za-z]+)\\.",

		)

	extract_var = col2.selectbox(
			"Extract From:",
			cat_var,
			key="extract_var"
		)

	if col2.button("Show Sample", key="creation_show_sample") and regex:
		crt = creator.Creator("Extract String", column=var, extract_col=extract_var, regex_pattern=regex)
		new_value = crt.fit_transform(data)
		st.dataframe(new_value.head())

	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name', key="temp_name")
	
	if st.button("Submit", key="extract_submit"):
		if var:
			crt = creator.Creator("Extract String", column=var, extract_col=extract_var, regex_pattern=regex)
			new_value = crt.fit_transform(data)

			if add_pipeline:
				name = f"{add_or_mod} column {var}"
				utils.add_pipeline(name, crt)

			utils.update_value(data_opt, new_value,temp_name,save_as)
			st.success("Success")

			utils.rerun()
			
		else:
			st.warning("New column name cannot be empty!")

def group_categorical(data, data_opt, var, add_pipeline, add_or_mod):
	temp_name=''
	columns = utils.get_variables(data)
	group_dict = {}

	col1, col2, col3, col4 = st.columns([1.7, 4, 2, 2.3])
	n_groups = col1.number_input(
			"N Group",
			1, 100, 3, 1,
			format="%d",
			key="n_groups"
		)

	group_var = col2.selectbox(
			"Group Column",
			columns,
			key="group_cat_var"
		)

	col3.markdown("#")
	sort_values = col3.checkbox("Sort Values", True, key="group_sort_values")

	col4.markdown("#")
	show_group = col4.checkbox("Show Group", key="group_show_group")

	unique_val = data[group_var].unique()
	if sort_values:
		unique_val = sorted(unique_val, key=lambda val: (val is np.nan, val))

	n_iter = 1 if (n_groups < 2) else int(n_groups-1)

	col1, col2 = st.columns([2.5,7.5])
	for i in range(n_iter):
		group_name = col1.text_input("Group Name", key=f"group_name_{i}")
		group_members = col2.multiselect("Group Members", unique_val, key=f"group_members_{i}")
		group_dict[group_name] = group_members

		# update unique value when selected to prevent same value in different group
		selected = [item for sublist in list(group_dict.values()) for item in sublist]
		unique_val = [val for val in unique_val if val not in selected]

	if n_groups > 1:
		col1, col2, col3 = st.columns([2.5,6.3,1.2])
		col3.markdown("#")
		if col3.checkbox("Other", key="group_other"):
			group_name = col1.text_input("Group Name", key="group_name_end")
			group_members = col2.multiselect("Group Members", unique_val, key=f"group_members_end", disabled=True)

			group_member_vals = sum(group_dict.values(), []) # Gather all group member values in 1D list
			group_members = [val for val in unique_val if val not in group_member_vals]
			group_dict[group_name] = group_members

		else:
			group_name = col1.text_input("Group Name", key="group_name_end")
			group_members = col2.multiselect("Group Members", unique_val, key=f"group_members_end")
			group_dict[group_name] = group_members

	col1, col2 = st.columns([2.5,7.5])
	if show_group:
		col2.write(group_dict)

	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name', key='creation')

	if col1.button("Submit", key="group_submit"):
		if var:
			crt = creator.Creator("Group Categorical", column=var, group_col=group_var, group_dict=group_dict)
			new_value = crt.fit_transform(data)
			
			if add_pipeline:
				name = f"{add_or_mod} column {var}"
				utils.add_pipeline(name, crt)

			utils.update_value(data_opt, new_value,temp_name,save_as)
			st.success("Success")

			utils.rerun()

		else:
			st.warning("New column name cannot be empty!")

def group_numerical(data, data_opt, var, add_pipeline, add_or_mod):
	temp_name=''
	num_var = utils.get_numerical(data)

	col1, col2, col3 = st.columns(3)
	n_groups = col1.number_input(
			"N Groups",
			1, 100, 2,
			key="n_bins"
		)

	group_var = col2.selectbox(
			"Bin Column",
			num_var,
			key="group_num_var"
		)

	col3.markdown("#")
	show_group = col3.checkbox("Show Bin Dict")

	group_dict = {}
	min_val, max_val = data[group_var].min(), data[group_var].max()
	for i in range(int(n_groups)):
		col1, col2, col3, col4 = st.columns([2.6,2.6,2.2,2.6])

		group_val = col4.number_input(
				f"Bin Value",
				i, 100, i,
				key=f"bin_value_{i}"
			)

		col3.markdown("#")
		use_operator = col3.checkbox("Use Operator", key=f"bin_use_operator_{i}")

		if use_operator:
			val1 = col1.selectbox(
					"Operator",
					["==", "!=", "<", ">", "<=", ">="],
					key=f"bin_operator_{i}"
				)

			val2 = col2.number_input(
					"Value",
					min_val, max_val, max_val,
					key=f"max_value_{i}"
				)

		else:
			val1 = col1.number_input(
				"Min Value",
				min_val, max_val, min_val,
				key=f"min_value_{i}"
			)

			val2 = col2.number_input(
					"Max Value",
					min_val, max_val, max_val,
					key=f"max_value_{i}"
				)

		group_dict[group_val] = (val1, val2)

	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name', key='create_data')

	if st.button("Submit"):
		if var:
			crt = creator.Creator("Group Numerical", column=var, group_col=group_var, group_dict=group_dict)
			new_value = crt.fit_transform(data)
			
			if add_pipeline:
				name = f"{add_or_mod} column {var}"
				utils.add_pipeline(name, crt)

			utils.update_value(data_opt, new_value,temp_name,save_as)
			st.success("Success")

			utils.rerun()

		else:
			st.warning("New column name cannot be empty!")

	if show_group :
		st.json(group_dict)

def replace_values(data,data_opt,var,add_pipeline):
	temp_name=''
	temp = data.copy(deep=True)
	column=st.session_state.modify_column_name
	new_value_input = st.selectbox('New Values', ['Text Input', 'Numpy Operations','Fill Null','String Operations'])
	if new_value_input!='Fill Null':
		if new_value_input == 'Text Input':
			li = []
			for i in temp[column]:
				if not isinstance(i, str) and not np.isnan(i):
					li.append(i)
				elif isinstance(i, str):
					li.append(i)

			old_value = st.selectbox('Old Value',set(li))
			new_value = st.text_input('New Value', np.nan)
		elif new_value_input=='Numpy Operations':
			new_value = st.selectbox('Select an operation:', [
				'np.log10',
				'np.sin',
				'np.cos',
				'np.tan',
				'np.exp'
			])
		else:
			options = {
				'Uppercase': str.upper,
				'Lowercase': str.lower,
				'Title case': str.title,
				'Strip leading/trailing whitespace': str.strip,
				'Remove leading whitespace': str.lstrip,
				'Remove trailing whitespace': str.rstrip,
				# 'Replace substring with another string': str.replace,
				'Remove characters': lambda s, char: s.replace(char, ''),
				# 'Split text into multiple columns': lambda s: s.str.split(expand=True),
			}

			# Create the select box in Streamlit
			operation = st.selectbox('Select an operation:', list(options.keys()))
			if operation=='Remove characters':
				char_to_remove= st.text_input('Enter character to remove:', '')

			# Apply the selected operation to the selected column of the dataframe
	else:
		# Select method to fill null values
		fill_method = st.selectbox(
			"Select method to fill null values:",
			["Custom Value", "Mean", "Median", "Mode", "From Another Column"],
		)
		if fill_method == "Custom Value":
			custom_value = st.text_input("Enter custom value")
		elif fill_method == "From Another Column":
			column_to_use = st.selectbox("Select column to use:", temp.columns)



	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name', key='creation')

	if st.button('submit', key='replace', type='primary'):
		if new_value_input == 'Text Input':
			temp[column] = temp[column].replace(old_value, new_value)
		elif new_value_input=='Numpy Operations':
			temp[column] = temp[column].astype('float').apply(eval(new_value))
		elif new_value_input=='Fill Null':
			# Fill null values based on selected method
			if fill_method == "Custom Value":
				temp[column] = temp[column].fillna(custom_value)
			elif fill_method == "Mean":
				temp[column] = temp[column].fillna(temp[column].mean())
			elif fill_method == "Median":
				temp[column] = temp[column].fillna(temp[column].median())
			elif fill_method == "Mode":
				temp[column] = temp[column].fillna(temp[column].mode().iloc[0])
			elif fill_method == "From Another Column":
				temp[column] = temp[column].fillna(temp[column_to_use])
		else:
			if 'Remove characters' in operation:
				temp[column] = temp[column].apply(options[operation], char=char_to_remove)
			else:
				temp[column] = temp[column].apply(options[operation])

		utils.update_value(data_opt, temp,temp_name,save_as)
		st.success("Success")
		utils.rerun()


from tqdm import tqdm
from epsilon import features
tqdm.pandas()
from rdkit import Chem

def my_progress_apply(data,data_opt,var,add_pipeline):
	temp_name = ''
	temp = data.copy(deep=True)
	col2,cx, col3 = st.columns([4,2, 2])
	if col3.button("Show Sample", key="creation_show_sample"):
		pass

	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name', key="temp_name")
	fun=[
		'Compute All Features using RDKit',
		'Chem.inchi.MolToInchiKey']
	selected_fun=st.selectbox('Select Function',fun)
	st.write('#')

	if st.button("Submit", key="progress_apply"):
		try:
			if selected_fun==fun[0]:
				new_value=temp.join(temp[var].progress_apply(features.ComputeAllFeatures).apply(
			lambda x: pd.Series(x, dtype='object')))
			# elif selected_fun==fun[1]:
			### 	problem is this is a list
			# 	new_value=temp[var].progress_apply(lambda x: Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(x))).to_list()
			utils.update_value(data_opt, new_value, temp_name, save_as)
			utils.rerun()
		except Exception as e:
			st.write(e)

def add_new(data,data_opt,var,add_pipeline):
	temp_name = ''
	temp = data.copy(deep=True)
	col2,cx, col3 = st.columns([4,2, 2])
	if col3.button("Show Sample", key="creation_show_sample"):
		pass
	slt_=st.radio('Select Method',['Input String','Copy Another Field'])
	if slt_=='Input String':
		value=st.text_input('Input String')
	else:
		col_name=st.selectbox('Select Field',temp.columns)


	col1, col2, c0 = st.columns([2, 2, 2])
	save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")

	if save_as:
		temp_name = col2.text_input('New Dataset Name', key="temp_name")


	if st.button("Submit", key="add_new"):
		if slt_ == 'Input String':
			temp[var]=value
		else:
			temp[var]=temp[col_name]
		utils.update_value(data_opt, temp, temp_name, save_as)
		utils.rerun()

