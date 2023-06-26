import streamlit as st
import numpy as np
from django.http import JsonResponse

from ...modules import utils
from ...modules.classes import dropper

def dropping(file ):
	data=file.get("data")
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

	option =file.get("default_columns")

	drop_var = file.get("select_columns")

	add_pipeline = file.get("add_to_pipeline")

	if drop_var:
		drp = dropper.Dropper(drop_var)
		new_value = drp.fit_transform(data)

		new_value = new_value.to_dict(orient="records")
		return JsonResponse(new_value, safe=False)

def drop_raw(data,file):

	option_ = [
		"With Null"
	]

	option = file.get("default_columns")

	# add_pipeline = file.get("Add To Pipeline", True, key="drop_add_row_pipeline")

	drop_var = file.get("select_columns")
	# remove all rows with null values
	# data = data.dropna()
	#
	# # remove all rows with blank values
	# data = data.replace('', np.nan).dropna()
	#
	# remove all rows with null or blank values in specific columns

	if drop_var:
		new_value = data.dropna(subset=drop_var)
		new_value = new_value.to_dict(orient="records")
		return JsonResponse(new_value, safe=False)

	#
	# # remove all rows with null or blank values in categorical columns
	# data = data.dropna(subset=['categorical_column']).replace('', np.nan).dropna(subset=['another_categorical_column'])
	#
	# # remove all rows with null values in specific columns and fill null values with mean or median
	# data['column1'].fillna(data['column1'].mean(), inplace=True)
	# data['column2'].fillna(data['column2'].median(), inplace=True)
