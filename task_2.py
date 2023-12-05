import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Set the environment variable to suppress joblib warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust the value based on the number of cores on your machine

# Load the data from the CSV file
df = pd.read_csv('Mall_customers.csv')

# Select relevant features for clustering
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_features = ['Gender']

# Preprocess numerical features (standardization)
numerical_transformer = StandardScaler()

# Preprocess categorical features (one-hot encoding)
categorical_transformer = OneHotEncoder()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to features
X = preprocessor.fit_transform(df)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # WCSS stands for "Within-Cluster Sum of Squares"
plt.show()

# Based on the Elbow method, choose the optimal number of clusters (k)
optimal_k = 5

# Apply K-means clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Display the resulting clusters
print(df[['CustomerID', 'Cluster']])

# Save the DataFrame with cluster assignments to a CSV file
df[['CustomerID', 'Cluster']].to_csv('customer_clusters.csv', index=False)

# Optional: Visualize the clusters in a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis', s=50)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-means Clustering')
plt.show()
