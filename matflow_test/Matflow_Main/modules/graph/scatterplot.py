import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
def Scatterplot(data,x,y,hue,title):
	if x != "-" and y != "-":
		fig, ax = plt.subplots()
		if len(title):
			ax.set_title(title)
		hue = None if (hue == "-") else hue
		ax = sns.scatterplot(data=data, x=x, y=y, hue=hue)
		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)
		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response