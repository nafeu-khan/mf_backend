import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse
def Violinplot(data,cat,num,hue,orient,dodge,split,title):
	if num != "-":
			fig, ax = plt.subplots()

			cat = None if (cat == "-") else cat
			hue = None if (hue == "-") else hue
			if cat == hue:
				split = False 

			if len(title)>0:
				# title = f"Violinplot of {num} by {cat}"
				#
				# if hue:
				# 	title = f"Violinplot of {num} by {cat} and {hue}"
				ax.set_title(title)

			if orient == "Vertical":
				if cat:
					ax = sns.violinplot(data=data, x=cat, y=num, hue=hue, dodge=dodge, split=split)
				else:
					ax = sns.violinplot(data=data, y=num, hue=hue, dodge=dodge, split=split)
			else:
				if cat:
					data[cat] = [str(cat) for cat in data[cat]]
					ax = sns.violinplot(data=data, x=num, y=cat, hue=hue, dodge=dodge, split=split)
				else:
					ax = sns.violinplot(data=data, x=num, hue=hue, dodge=dodge, split=split)

			image_stream = io.BytesIO()
			plt.savefig(image_stream, format='png')
			plt.close(fig)
			image_stream.seek(0)
			response = HttpResponse(content_type='image/png')
			response.write(image_stream.getvalue())
			return response
	return HttpResponse("Invalid parameters or method.", status=400)
