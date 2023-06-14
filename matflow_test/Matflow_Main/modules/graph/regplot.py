import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
def Regplot(data,x,y,title,sctr):
	if x != "-" and y != "-":
		fig, ax = plt.subplots()
		if len(title):
			ax.set_title(title)
		ax = sns.regplot(data=data, x=x, y=y, scatter=sctr)
		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)
		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response