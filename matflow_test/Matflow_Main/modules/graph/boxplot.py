import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse

def boxplot(data,title,cat,num,hue,orient,dodge):

	if num != "-":
		fig, ax = plt.subplots()

		cat = None if (cat == "-") else cat
		hue = None if (hue == "-") else hue

		if title.length>0:
			default_title = f"Boxplot of {num} by {cat}"

			if hue:
				default_title = f"Boxplot of {num} by {cat} and {hue}"
			ax.set_title(title)

		if orient == "Vertical":
			if cat:
				ax = sns.boxplot(data=data, x=cat, y=num, hue=hue, dodge=dodge)
			else:
				ax = sns.boxplot(data=data, y=num, hue=hue, dodge=dodge)
		else:
			if cat:
				data[cat] = [str(cat) for cat in data[cat]]
				ax = sns.boxplot(data=data, x=num, y=cat, hue=hue, dodge=dodge)
			else:
				ax = sns.boxplot(data=data, x=num, hue=hue, dodge=dodge)
		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)
		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response
	return HttpResponse("Invalid parameters or method.", status=400)

