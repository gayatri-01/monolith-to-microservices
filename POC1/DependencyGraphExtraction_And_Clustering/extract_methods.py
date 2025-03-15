import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Path to Java source code
PROJECT_PATH = "../Monolith/src/main/java/com/monolith/poc/"

# Regular expressions
METHOD_REGEX = re.compile(r"public\s+\w+\s+(\w+)\(")  # Extracts method names

# Extract method names from all Java files
class_methods = {}

def extract_methods():
    """Extract method names from Java files."""
    for root, _, files in os.walk(PROJECT_PATH):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract class name
                class_match = re.search(r"public\s+class\s+(\w+)", content)
                if class_match:
                    class_name = class_match.group(1)
                    method_names = METHOD_REGEX.findall(content)

                    if method_names:
                        class_methods[class_name] = " ".join(method_names)  # Combine all methods into a single string

extract_methods()



# Convert method names into TF-IDF vectors
vectorizer = TfidfVectorizer()
method_vectors = vectorizer.fit_transform(class_methods.values())

# Apply KMeans Clustering (Finding optimal number of clusters)
num_clusters = 2 # Adjust based on silhouette score or elbow method
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(method_vectors)

# Assign clusters to classes
microservices = {}
for i, class_name in enumerate(class_methods.keys()):
    cluster_id = labels[i]
    if cluster_id not in microservices:
        microservices[cluster_id] = []
    microservices[cluster_id].append(class_name)

# Display microservices
print("\n🔹 Suggested Microservices Based on Functional Similarity:")
for service, classes in microservices.items():
    print(f"  ➤ Microservice {service + 1}: {classes}")

# Visualize using a simple graph
graph = nx.Graph()
for service, classes in microservices.items():
    for class_name in classes:
        graph.add_node(class_name, group=service)

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(graph)
colors = [labels[i] for i, _ in enumerate(class_methods.keys())]
nx.draw(graph, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3)
plt.title("Functional Similarity-Based Microservices Clustering")
plt.show()
