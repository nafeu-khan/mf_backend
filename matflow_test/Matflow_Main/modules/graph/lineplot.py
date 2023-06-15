import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
def Lineplot(data,x,y,hue,title,style,legend):
	if x != "-" and y != "-":
		fig, ax = plt.subplots()
		hue = None if (hue == "-") else hue
		style = None if (style == "") else style
		if len(title)>0:
			ax.set_title(title)
		ax = sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, legend=legend)	
		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)
		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response