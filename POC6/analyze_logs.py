import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from datetime import datetime
from networkx.algorithms import community

LOG_FILE = "application.log"  # Path to the log file

# Regular expressions to extract log details
log_patterns = {
    "account_creation": re.compile(r"Account created with ID: (\d+)"),
    "transaction": re.compile(r"Transaction recorded: Transfer from (\d+) to (\d+) of amount \$(\d+\.?\d*)"),
    "loan": re.compile(r"Loan of \$(\d+\.?\d*) applied for account: (\d+)"),
    "customer_registration": re.compile(r"Customer registered with ID: (\d+)"),
    "notification": re.compile(r"Notification sent: (.+)")
}

def parse_timestamp(line):
    match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return None

def visualize_service_dependencies(service_dependencies):
    G = nx.DiGraph()
    
    for (service_a, service_b, weight) in service_dependencies:
        G.add_edge(service_a, service_b, weight=weight)
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(a, b): f"{w}" for a, b, w in service_dependencies})
    
    plt.title("Service Dependency Graph")
    plt.show()

import networkx as nx
import networkx.algorithms.community as community
from sklearn.metrics import silhouette_score
import numpy as np
import community.community_louvain as community_louvain

def detect_service_clusters(service_dependencies):
    G = nx.Graph()
    
    for (service_a, service_b, weight) in service_dependencies:
        if weight > 5:  # Ignore weak dependencies
            G.add_edge(service_a, service_b, weight=weight)
    
    communities_generator = community.louvain_communities(G, weight='weight', resolution=1.8)
    
    clusters = {f"Cluster {i+1}": list(c) for i, c in enumerate(communities_generator)}
    
    print("Detected Microservice Clusters:")
    for cluster, services in clusters.items():
        print(f"{cluster}: {services}")


    #partition = community_louvain.best_partition(G.to_undirected()) 

    # Convert clusters to partition format for modularity
    partition = {node: i for i, cluster in enumerate(communities_generator) for node in cluster}

    # Compute modularity
    modularity = community_louvain.modularity(community_louvain.best_partition(G.to_undirected()), G.to_undirected())
    #modularity = nx.algorithms.community.quality.modularity(G, G.to_undirected())
    normalized_modularity = (modularity + 0.5) / 1.5  # Scale to [0,1]

    print(f"Modularity Score: {normalized_modularity:.4f}")

    # Generate labels from partition
    labels = np.array([partition[node] for node in G.nodes])

    # Assume service_embeddings is a NumPy array of shape (num_services, embedding_dim)

    adj_matrix = nx.to_numpy_array(G)
    node_index_map = {node: i for i, node in enumerate(G.nodes())}
    labels = np.array([partition[node] for node in G.nodes()])

    silhouette = silhouette_score(adj_matrix, labels, metric='euclidean')
    print(f"Silhouette Score: {silhouette:.4f}")

    total_edges = G.number_of_edges()
    inter_service_edges = sum(1 for edge in G.edges if partition[edge[0]] != partition[edge[1]])

    coupling = inter_service_edges / total_edges if total_edges > 0 else 0
    print(f"Service Coupling Score: {coupling:.4f}")

    intra_service_edges = sum(1 for edge in G.edges if partition[edge[0]] == partition[edge[1]])

    cohesion = intra_service_edges / total_edges if total_edges > 0 else 0
    print(f"Service Cohesion Score: {cohesion:.4f}")



    
    return clusters

def analyze_logs():
    service_usage = defaultdict(int)
    service_dependencies = defaultdict(int)
    user_sessions = []
    
    session_activity = defaultdict(list)
    recent_service_calls = defaultdict(lambda: deque(maxlen=3))  # Track last 3 services per user
    
    with open(LOG_FILE, "r") as file:
        for line in file:
            matched_services = []
            timestamp = parse_timestamp(line)
            
            for service, pattern in log_patterns.items():
                if pattern.search(line):
                    service_usage[service] += 1
                    matched_services.append(service)
            
            user_id_match = re.search(r"User: (\d+)", line)
            if user_id_match:
                user_id = user_id_match.group(1)
                session_activity[user_id].extend(matched_services)
                
                # Track dependencies based on last 3 calls
                for service in matched_services:
                    for prev_service in recent_service_calls[user_id]:
                        if prev_service != service:
                            service_dependencies[(prev_service, service)] += 1
                    recent_service_calls[user_id].append(service)
    
    # Convert session activity into user session lists
    for user_id, services in session_activity.items():
        user_sessions.append(list(set(services)))
    
    print("Service Usage Statistics:")
    for service, count in service_usage.items():
        print(f"{service}: {count} logs")
    
    print("Service Dependencies:")
    filtered_dependencies = [(a, b, w) for (a, b), w in service_dependencies.items() if w > 2]  # Filter weak links
    for service_a, service_b, weight in filtered_dependencies:
        print(f"{service_a} -> {service_b} (Weight: {weight})")
    
    # Visualize service dependencies
    visualize_service_dependencies(filtered_dependencies)
    
    # Detect and print clusters
    clusters = detect_service_clusters(filtered_dependencies)
    
    return service_usage, service_dependencies, user_sessions, clusters

if __name__ == "__main__":
    analyze_logs()
