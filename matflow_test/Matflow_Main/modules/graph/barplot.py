import io
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponse


def Barplot(data,cat,num,hue,orient,annotate,title):
	errorbar=True
	fig, ax = plt.subplots()
	if cat != "-" and num != "-":
		hue = None if (hue == "-") else hue
		errorbar = ("ci", 95) if errorbar== True else None

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
		if len(title)> 0:
			# title = f"Barplot of {num} by {cat}"
			# if hue:
			# 	title = f"Barplot of {num} by {cat} and {hue}"
			ax.set_title(title)
		if annotate== True:
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

		image_stream = io.BytesIO()
		plt.savefig(image_stream, format='png')
		plt.close(fig)
		image_stream.seek(0)
		response = HttpResponse(content_type='image/png')
		response.write(image_stream.getvalue())
		return response

	return HttpResponse("Invalid parameters or method.", status=400)


