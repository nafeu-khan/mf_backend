import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
from io import BytesIO
from ...modules import utils

def barplot(data,request):
	num_var = utils.get_numerical(data, add_hypen=True)
	low_cardinality = utils.get_low_cardinality(data, add_hypen=True)

	cat = request.GET.get('cat')  # Get the categorical variable from the query parameter
	num = request.GET.get('num')  # Get the numerical variable from the query parameter
	hue = request.GET.get('hue')  # Get the hue variable from the query parameter
	orient = request.GET.get('orient', 'Vertical')  # Get the orientation from the query parameter

	fig, ax = plt.subplots()

	if cat != "-" and num != "-":
		hue = None if (hue == "-") else hue
		errorbar = ("ci", 95) if request.GET.get('errorbar') == 'true' else None

		if request.GET.get('set_title') == 'true':
			title = request.GET.get('title', f"{num} by {cat}")
			ax.set_title(title)

		if orient == "Vertical":
			try:
				data[cat] = data[cat].astype(int)
			except:
				pass
			ax = sns.barplot(data=data, x=cat, y=num, hue=hue, errorbar=errorbar)
		else:
			data[cat] = data[cat].astype(str)
			order = sorted(data[cat].unique())
			ax = sns.barplot(data=data, x=num, y=cat, hue=hue, order=order, errorbar=errorbar)

		if request.GET.get('annotate') == 'true':
			if orient == "Vertical":
				for bar in ax.patches:
					ax.annotate(format("{:.3f}".format(bar.get_height())),
								(bar.get_x() + 0.5 * bar.get_width(),
								 bar.get_height()), ha='center', va='center',
								size=11, xytext=(0, 8),
								textcoords='offset points'
								)
			else:
				for rect in ax.patches:
					plt.text(1.05 * rect.get_width(),
							 rect.get_y() + 0.5 * rect.get_height(),
							 '%.3f' % float(rect.get_width()),
							 ha='center', va='center'
							 )

	# Save the plot image to a BytesIO object
	image_stream = io.BytesIO()
	plt.savefig(image_stream, format='png')
	plt.close(fig)
	image_stream.seek(0)

	# Create a response with the image data
	response = HttpResponse(content_type='image/png')
	response.write(image_stream.getvalue())

	return response