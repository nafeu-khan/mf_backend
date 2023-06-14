import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse

def Histogram(data,var,title,hue,orient,stat,auto_bin,kde,legend):
	if auto_bin>0:
		bins=auto_bin
	else :
		bins = "auto"
	if var != "-":
		fig, ax = plt.subplots()
		hue = None if (hue == "-") else hue
		if len(title):
			ax.set_title(title)
		if var != "-":
			if orient == "Vertical":
				ax = sns.histplot(data=data, x=var, bins=bins, hue=hue, kde=kde, legend=legend, stat=stat)	
			else:
				ax = sns.histplot(data=data, y=var, bins=bins, hue=hue, kde=kde, legend=legend, stat=stat)	
			image_stream = io.BytesIO()
			plt.savefig(image_stream, format='png')
			plt.close(fig)
			image_stream.seek(0)
			response = HttpResponse(content_type='image/png')
			response.write(image_stream.getvalue())
			return response

