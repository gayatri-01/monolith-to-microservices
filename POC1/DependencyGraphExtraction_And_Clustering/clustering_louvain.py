import networkx as nx
import matplotlib.pyplot as plt
import extract_dependencies as ed
# import community as community_louvain  # Louvain Algorithm for clustering
import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score
import community.community_louvain as community_louvain



# Create a directed graph
dependency_graph = ed.extract_dependencies()

# **Step 1: Apply Louvain Clustering**
partition = community_louvain.best_partition(dependency_graph.to_undirected())  # Compute clusters dynamically

# **Step 2: Convert to microservices based on detected communities**
microservices = {}
for class_name, cluster_id in partition.items():
    if cluster_id not in microservices:
        microservices[cluster_id] = []
    microservices[cluster_id].append(class_name)

# **Step 3: Display results**
print("\nSuggested Microservices:")
for service, classes in microservices.items():
    print(f"Microservice {service + 1}: {classes}")

# **Step 4: Visualize Clusters**
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(dependency_graph)
colors = [partition[node] for node in dependency_graph.nodes()]
nx.draw(dependency_graph, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3, edge_color="gray")
plt.title("Louvain-Based Microservices Clustering")
plt.show()


# **Step 4: Compute Metrics**

## **1. Modularity Score**
modularity = community_louvain.modularity(partition, dependency_graph.to_undirected())

## **2. Silhouette Score**
# Create adjacency matrix for distance-based silhouette calculation
adj_matrix = nx.to_numpy_array(dependency_graph)
node_index_map = {node: i for i, node in enumerate(dependency_graph.nodes())}
labels = np.array([partition[node] for node in dependency_graph.nodes()])

silhouette = silhouette_score(adj_matrix, labels, metric='euclidean')

## **3. Service Coupling**
total_edges = len(dependency_graph.edges)
inter_service_edges = sum(
    1 for u, v in dependency_graph.edges if partition[u] != partition[v]
)
service_coupling = inter_service_edges / total_edges if total_edges > 0 else 0

## **4. Service Cohesion**
intra_service_edges = sum(
    1 for u, v in dependency_graph.edges if partition[u] == partition[v]
)
service_cohesion = intra_service_edges / total_edges if total_edges > 0 else 0

# # **Step 5: Print Results**
# print("\n🔹 Suggested Microservices:")
# for service, classes in microservices.items():
#     print(f"  ➤ Microservice {service + 1}: {classes}")

print("\n**Clustering Evaluation Metrics**")
print(f"Modularity Score: {modularity:.4f} (Higher is better)")
print(f"Silhouette Score: {silhouette:.4f} (Higher is better)")
print(f"Service Coupling: {service_coupling:.4f} (Lower is better)")
print(f"Service Cohesion: {service_cohesion:.4f} (Higher is better)")