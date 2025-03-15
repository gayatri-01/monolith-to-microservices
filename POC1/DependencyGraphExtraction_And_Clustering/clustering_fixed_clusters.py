import numpy as np
import extract_dependencies as ed
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Create a directed graph
dependency_graph = ed.extract_dependencies()

# Convert graph to adjacency matrix
adj_matrix = nx.to_numpy_array(dependency_graph)

# Apply Spectral Clustering
num_clusters = 5  # Assuming we want 5 microservices
clustering = SpectralClustering(n_clusters=num_clusters, affinity="precomputed", assign_labels="kmeans")
labels = clustering.fit_predict(adj_matrix)

# Assign cluster labels to classes
class_clusters = {list(dependency_graph.nodes())[i]: labels[i] for i in range(len(labels))}

# Display Microservice Grouping
microservices = {}
for class_name, cluster_id in class_clusters.items():
    if cluster_id not in microservices:
        microservices[cluster_id] = []
    microservices[cluster_id].append(class_name)

for service, classes in microservices.items():
    print(f"Microservice {service + 1}: {classes}")

# Step 5: Create a visualization
plt.figure(figsize=(10, 7))

# Generate layout for better visualization
pos = nx.spring_layout(dependency_graph, seed=42)  

# Assign colors based on clusters
node_colors = [labels[i] for i in range(len(labels))]

# Draw the network with clustered colors
nx.draw(dependency_graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Set1, node_size=700, edge_color="gray")

plt.title("Spectral Clustering of Call Graph into Microservices")
plt.show()    