import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


csv_path = "/home/feketeova/Documents/results.csv"
df = pd.read_csv(csv_path, header=0)

# removing title, id, and headings
df = df.iloc[:, 1:]
df = df.iloc[1:, :]
numeric_df = df.select_dtypes(include=[np.number])

# Go feature by feature
for col in numeric_df.columns:

    data = numeric_df[col].dropna().values

   #removing outliers
    z_scores = np.abs(zscore(data))
    data_clean = data[z_scores < 3]



    #normalize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clean.reshape(-1, 1))  # shape (n,1)

    #clustering
    linkage_data = linkage(data_scaled, method='average', metric='euclidean')

    #dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_data)
    plt.title(f"Patient clustering for feature: {col}")
    plt.ylabel("Euclidean distance")
    plt.show()