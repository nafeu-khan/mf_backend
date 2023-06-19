import streamlit as st
import pandas as pd
import numpy as np
from django.http import JsonResponse
from ...modules import utils
from ...modules.classes import creator

def creation(file):
	# variables = utils.get_variables(data)
	# col1, col2, col3, col4 = st.columns([1.6, 3, 2.4, 2.4])
	data=file.get("file")
	data=pd.DataFrame(data)
	add_or_mod =file.get("option")
	# st.session_state.add=add_or_mod=='Add'
	method_name = ["New Column","Math Operation", "Extract Text", "Group Categorical", "Group Numerical"]
	if add_or_mod == "Modify":
		method_name.pop(0)
		method_name.append("Replace Values")
		method_name.append("Progress Apply")
	# if add_or_mod == "Add":
	var = file.get("column_name")
	# print (f"var={var}")
	# else:
	# 	var = file.get("select_column")
	add_pipeline = file.get("add_to_pipeline")
	method = file.get("method")
	# var = var.strip() # remove whitespace

	if method == "Math Operation":
		return math_operation(data, var, add_pipeline, add_or_mod,file)
	elif method == "Extract Text":
		return extract_text(data, var, add_pipeline, add_or_mod,file)
	elif method == "Group Categorical":
		return group_categorical(data, var, add_pipeline, add_or_mod,file)
	elif method == "Group Numerical":
		return group_numerical(data, var, add_pipeline, add_or_mod,file)
	elif add_or_mod=="Modify" and method=="Replace Values":
		return replace_values(data,var,add_pipeline)
	elif method=='Progress Apply':
		return my_progress_apply(data,var,add_pipeline)
	elif method=="New Column":
		return add_new(data,var,add_pipeline,file)

def add_new(data,var,add_pipeline,file):
	temp = data.copy(deep=True)
	slt_=file.get("Select Method")
	if slt_=='Input String':
		value=file.get('Input String')
	else:
		col_name=file.get('Select Field')
	if slt_ == 'Input String':
		temp[var]=value
	else:
		temp[var]=temp[col_name]
	df = temp.to_dict(orient="records")
	return JsonResponse(df, safe=False)


def math_operation(data, var, add_pipeline, add_or_mod,file):
	operation =file.get("data").get("new_value_operation")
	print(f"op = {operation} var = {var}")
	crt = creator.Creator("Math Operation", var, operation_string=operation)
	new_value = crt.fit_transform(data)
	df=new_value.to_dict(orient="records")
	return JsonResponse(df,safe=False)
	# col1, col2,c0 = st.columns([2,2,2])
	# save_as = col1.checkbox('Save as New Dataset', True, key="save_as_new")
	#
	# if save_as:
	# 	temp_name = col2.text_input('New Dataset Name',key="temp_name")
	#
	# if st.button("Submit", key="math_submit"):
	# 	if var:
	# 		crt = creator.Creator("Math Operation", column=var, operation_string=operation)
	# 		new_value = crt.fit_transform(data)
	#
	# 		if add_pipeline:
	# 			name = f"{add_or_mod} column {var}"
	# 			utils.add_pipeline(name, crt)
	#
	# 		utils.update_value( new_value,temp_name,save_as)
	# 		st.success("Success")
	#
	# 		utils.rerun()
	#
	# 	else:
	# 		st.warning("New column name cannot be empty!")
def extract_text(data,  var, add_pipeline, add_or_mod,file):
	regex = file.get("regex")
	extract_var =file.get("extract_from")
	crt = creator.Creator("Extract String", column=var, extract_col=extract_var, regex_pattern=regex)
	new_value = crt.fit_transform(data)
	df = new_value.to_dict(orient="records")
	return JsonResponse(df, safe=False)

def group_categorical(data,  var, add_pipeline, add_or_mod,file):
	group_dict = {}
	n_groups = file.get("n_groups")
	group_var = file.get("group_cat_var")
	sort_values = file.get("group_sort_values")
	show_group = file.get("group_show_group")

	unique_val = data[group_var].unique()
	if sort_values:
		unique_val = sorted(unique_val, key=lambda val: (val is np.nan, val))

	n_iter = 1 if (n_groups < 2) else int(n_groups-1)

	for i in range(n_iter):
		group_name = file.get("group_name")
		group_members = file.get("group_members")
		group_dict[group_name] = group_members
		selected = [item for sublist in list(group_dict.values()) for item in sublist]
		unique_val = [val for val in unique_val if val not in selected]

	if n_groups > 1:
		OTHER=file.get("other")
		if OTHER==True:
			group_name = file.get("group_name_end")
			group_members = file.get("group_members_end")
			group_member_vals = sum(group_dict.values(), []) # Gather all group member values in 1D list
			group_members = [val for val in unique_val if val not in group_member_vals]
			group_dict[group_name] = group_members

		else:
			group_name = file.get("group_name_end")
			group_members = file.get("group_members_end")
			group_dict[group_name] = group_members
	# if show_group:
	# 	col2.write(group_dict)
	crt = creator.Creator("Group Categorical", column=var, group_col=group_var, group_dict=group_dict)
	new_value = crt.fit_transform(data)
	new_value=new_value.to_dict(orient="records")
	return JsonResponse(new_value,safe=False)

def group_numerical(data,  var, add_pipeline, add_or_mod,file):
	temp_name=''
	num_var = utils.get_numerical(data)

	col1, col2, col3 = st.columns(3)
	n_groups = file.get("n_bins")

	group_var = file.get("group_num_var")

	show_group = file.get("Show Bin Dict")

	group_dict = {}
	min_val, max_val = data[group_var].min(), data[group_var].max()
	for i in range(int(n_groups)):
		group_val = file.get("bin_value")
		use_operator = file.get("bin_use_operator")
		if use_operator:
			val1 = file.get("bin_operator")
			val2 = file.get("max_value")
		else:
			val1 = file.get("min_value")
			val2 = file.get("max_value")
		group_dict[group_val] = (val1, val2)
	crt = creator.Creator("Group Numerical", column=var, group_col=group_var, group_dict=group_dict)
	new_value = crt.fit_transform(data)
	new_value = new_value.to_dict(orient="records")
	return JsonResponse(new_value, safe=False)

def replace_values(data,var,add_pipeline):
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

		utils.update_value( temp,temp_name,save_as)
		st.success("Success")
		utils.rerun()
from tqdm import tqdm
from ...epsilon import features
tqdm.pandas()
from rdkit import Chem
def my_progress_apply(data,var,add_pipeline):
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
			utils.update_value( new_value, temp_name, save_as)
			utils.rerun()
		except Exception as e:
			st.write(e)

