import io
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
from ...modules import utils

def Pieplot(data,var,explode,set_title,label,pct):
	# low_cardinality = utils.get_low_cardinality(data, add_hypen=True)
	#
	# col1, col2 = st.columns(2)
	# var = col1.selectbox(
	# 		"Variable",
	# 		low_cardinality,
	# 		key="pie_var"
	# 	)
	#
	# explode = col2.number_input(
	# 	"	Explode value (0 - 0.1)",
	# 		0.0, 0.1, 0.0, 0.01, # min, max, default, step
	# 		key="pie_explode"
	# 	)

	# col1, col2, col3, col4, _ = st.columns([1.5, 1.5, 1.8, 1.5, 3.7])
	# set_title = col1.checkbox("Title", key="pie_set_title")
	# label = col2.checkbox("Label", True, key="pie_label")
	# pct = col3.checkbox("Percentage", True, key="pie_pct")
	# shadow = col4.checkbox("Shadow", key="pie_shadow[1.5, 1.5, 7]")

	if var != "-":
		fig, ax = plt.subplots()

		if set_title.length>0:
			ax.set_title(set_title)

		pct = '%1.2f%%' if pct else None

		ax = data[var].value_counts().plot(kind="pie", 
				explode=[explode for x in data[var].dropna().unique()], 
				autopct=pct
			)
			
		if label.length>0:
			ax.set_ylabel(var)
		else:
			ax.set_ylabel("")

		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)

		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response
	return HttpResponse("Invalid parameters or method.", status=400)

