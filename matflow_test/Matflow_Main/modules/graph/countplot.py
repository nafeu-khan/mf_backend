import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse

def Countplot(data,var,title,hue,orient,annot):
	if var != "-":
		fig, ax = plt.subplots()
		if hue == "-":
			hue = None
		if len(title) > 0:
			# title = f"{var} Count"
			ax.set_title(title)
			ax.title.set_position([.5, 1.5])

		if orient == "Vertical":
			ax = sns.countplot(data=data, x=var, hue=hue)
		else:
			ax = sns.countplot(data=data, y=var, hue=hue)

		if annot:
			if orient == "Vertical":
				for bar in ax.patches:
					ax.annotate(format(int(bar.get_height())),
				            (bar.get_x()+0.5*bar.get_width(),
				            bar.get_height()), ha='center', va='center',
				            size=11, xytext=(0, 8),
				            textcoords='offset points'
			            )
			else:
				for rect in ax.patches:
					plt.text(1.05*rect.get_width(), 
							rect.get_y()+0.5*rect.get_height(),
				            '%d' % int(rect.get_width()),
				            ha='center', va='center'
			            )
		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)
		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response
	return HttpResponse("Invalid parameters or method.", status=400)

