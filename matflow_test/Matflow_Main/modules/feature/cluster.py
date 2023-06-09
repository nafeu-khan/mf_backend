import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_dataset(data):
    global fig
    try:
        df = data
        col, col2 = st.columns(2)
        n_cls = col.slider(
            f"Number of classes",
            0, 10, 3,
            key="number_of_classes"
        )
        class_nms = []
        c=0
        for i in range(n_cls):
            if c==0 :
                col,col1=st.columns(2)
            if(i%2==0):
                class_nm = col.text_input(f"Class {i+1} Name",f"Class {i+1}")
            else :
                class_nm = col1.text_input(f"Class {i+1} Name",f"Class {i+1}")
            class_nms.append(class_nm)
            c=c+1
            c=c%2

        col1, col2 = st.columns(2)
        display_type = col1.selectbox(
            "Display Type",
            ["Graph", "Table"]
        )
        if st.button("Submit", key="split_submit_button"):
            X = df.iloc[:, :-1].values
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            kmeans = KMeans(n_clusters=n_cls, random_state=0).fit(X)
            centroids = pca.transform(kmeans.cluster_centers_)

            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
            handles, labels = scatter.legend_elements()
            cluster_labels = class_nms      #[label]) for label in kmeans.labels_]
            centroid_labels = ["" for _ in range(n_cls)]
            ax.legend(handles, cluster_labels)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o')
            for i, txt in enumerate(centroid_labels):
                ax.annotate(txt, (centroids[i, 0], centroids[i, 1]), xytext=(-10, 10),
                            textcoords='offset points', color='red')

            ax.set_title('K-means Clustering of Dataset')

            df=data
            df['Class'] = [cluster_labels[label] for label in kmeans.labels_]


            if display_type == "Table":
                col2.markdown("#")
                st.table(df)
            else:
                st.pyplot(fig)
    except Exception as e:
        st.error(e)
