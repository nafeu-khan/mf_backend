import io
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.http import HttpResponse

def Boxplot(data, title, cat, num, hue, orient, dodge):
    if num != "-":
        cat = None if cat == "-" else cat
        hue = None if hue == "-" else hue
        fig, ax = plt.subplots()
        if len(title) > 0:
            # title = f"Boxplot of {num} by {cat}"
            # if hue:
            #     title = f"Boxplot of {num} by {cat} and {hue}"
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

        # Generate the plot as an image
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='png')
        plt.close(fig)
        image_stream.seek(0)

        # Return the image as a Django HttpResponse
        response = HttpResponse(content_type='image/png')
        response.write(image_stream.getvalue())
        return response
