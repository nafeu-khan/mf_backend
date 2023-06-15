import io
import matplotlib.pyplot as plt
from django.http import HttpResponse

def Pieplot(data,var,explode,title,label,pct):
	explode = float(explode)
	if var != "-":
		fig, ax = plt.subplots()
		if len(title) >0:
			# title = f"{var} Pie Plot"
			ax.set_title(title)
		pct = '%1.2f%%' if pct else None

		ax = data[var].value_counts().plot(kind="pie", 
				explode=[explode for x in data[var].dropna().unique()], 
				autopct=pct
			)
			
		if label:
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

