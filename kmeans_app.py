import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.cluster import KMeans
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ K-Means Customer Segmentation")
st.write("Upload a customer dataset to perform clustering based on Annual Income and Spending Score.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    # Select features
    if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

        # Elbow method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Elbow plot
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        k = st.slider("Select number of clusters (K)", 2, 10, 5)

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        df['Cluster'] = y_kmeans

        # Visualize clusters
        fig2, ax2 = plt.subplots()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'purple', 'orange', 'brown', 'pink']
        for i in range(k):
            ax2.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                        s=100, c=colors[i], label=f'Cluster {i}')
        ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    s=300, c='black', marker='X', label='Centroids')
        ax2.set_title('Customer Clusters')
        ax2.set_xlabel('Annual Income (k$)')
        ax2.set_ylabel('Spending Score (1-100)')
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Clustered Data")
        st.write(df.head())

        # Download clustered data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data", csv, "segmented_customers.csv", "text/csv")
    else:
        st.error("Dataset must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")
