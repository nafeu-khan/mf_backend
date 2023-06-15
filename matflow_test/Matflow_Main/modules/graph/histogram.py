import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from django.http import HttpResponse

def Histogram(data, var, title, hue, orient, stat, auto_bin, kde, legend):
    if auto_bin > 0:
        bins = auto_bin
    else:
        bins = "auto"

    if var != "-":
        fig, ax = plt.subplots()
        hue = None if hue == "-" else hue
        if len(title):
            ax.set_title(title)
        if var != "-":
            if orient == "Vertical":
                ax = sns.histplot(data=data, x=var, bins=bins, hue=hue, kde=kde, legend=legend, stat=stat)
            else:
                ax = sns.histplot(data=data, y=var, bins=bins, hue=hue, kde=kde, legend=legend, stat=stat)

            # Save the plot to a BytesIO stream
            image_stream = io.BytesIO()
            plt.savefig(image_stream, format='png')
            plt.close(fig)
            image_stream.seek(0)

            # Convert the matplotlib plot to Plotly graph object
            pio.templates.default = "plotly_dark"  # Use a dark theme, you can change this if desired
            graph = go.Figure(go.Image(source=image_stream.getvalue()))

            # Convert the graph to HTML and send as a response
            html_content = pio.to_html(graph, full_html=False)
            response = HttpResponse(content_type='text/html')
            response.write(html_content)
            return response
