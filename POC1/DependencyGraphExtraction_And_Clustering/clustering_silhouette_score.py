from sklearn.metrics import silhouette_score
import extract_dependencies as ed
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Create a directed graph
dependency_graph = ed.extract_dependencies()

# Convert graph to adjacency matrix
adj_matrix = nx.to_numpy_array(dependency_graph)

# Find the optimal number of clusters
best_k = None
best_score = -1
silhouette_scores = []

for k in range(2, min(10, len(dependency_graph.nodes()))):  # Ensure k ≤ number of nodes
    clustering = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=42)
    
    try:
        labels = clustering.fit_predict(adj_matrix)
        score = silhouette_score(adj_matrix, labels, metric="precomputed")
        silhouette_scores.append((k, score))
        
        if score > best_score:
            best_score = score
            best_k = k

        print(f"✅ k={k}, Silhouette Score: {score:.4f}")
    
    except Exception as e:
        print(f"❌ k={k} failed: {e}")

print(f"\n🔹 Optimal number of clusters: {best_k} with Silhouette Score {best_score:.4f}")

# Visualize Silhouette Scores
# plt.plot(*zip(*silhouette_scores), marker="o")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Scores for Different k")
# plt.show()

# Perform Clustering with Best k
if best_k:
    clustering = SpectralClustering(n_clusters=best_k, affinity="precomputed", assign_labels="kmeans", random_state=42)
    labels = clustering.fit_predict(adj_matrix)

    # Assign cluster labels to classes
    class_clusters = {list(dependency_graph.nodes())[i]: labels[i] for i in range(len(labels))}

    # Display Microservice Grouping
    microservices = {}
    for class_name, cluster_id in class_clusters.items():
        if cluster_id not in microservices:
            microservices[cluster_id] = []
        microservices[cluster_id].append(class_name)

    print("\n🔹 Suggested Microservices:")
    for service, classes in microservices.items():
        print(f"  ➤ Microservice {service + 1}: {classes}")


