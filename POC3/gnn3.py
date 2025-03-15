#KMeans with Custom Penalty for Resource Considerations
import json
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import json
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import numpy as np

import json
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Load graph from JSON
with open("D:\\Gayatri\\BITS WILP\\Dissertation\\Parser\\dayTraderGraph2.json", 'r') as f:
    graph = json.load(f)

# Identify orphan nodes
edge_nodes = set()
for edge in graph['edges']:
    edge_nodes.add(edge['from'])
    edge_nodes.add(edge['to'])
all_nodes_initial = [node['id'] for node in graph['nodes']]
orphan_nodes = [node for node in all_nodes_initial if node not in edge_nodes]

# Filter out orphan nodes from the graph
connected_nodes = [node for node in graph['nodes'] if node['id'] in edge_nodes]
connected_node_ids = {node['id'] for node in connected_nodes}
connected_edges = [edge for edge in graph['edges'] if edge['from'] in connected_node_ids and edge['to'] in connected_node_ids]

# Create node ID to index mapping for connected nodes
node_id_to_idx = {}
for node in connected_nodes:
    node_type = node['type']
    if node_type not in node_id_to_idx:
        node_id_to_idx[node_type] = {}
    node_id_to_idx[node_type][node['id']] = len(node_id_to_idx[node_type])

# Create HeteroData object for connected nodes
data = HeteroData()
for node_type in node_id_to_idx:
    data[node_type].num_nodes = len(node_id_to_idx[node_type])

# Add edges for connected nodes
edge_type_to_from_to = {}
for edge in connected_edges:
    from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
    from_type = next(node['type'] for node in connected_nodes if node['id'] == from_id)
    to_type = next(node['type'] for node in connected_nodes if node['id'] == to_id)
    if edge_type not in edge_type_to_from_to:
        edge_type_to_from_to[edge_type] = {}
    if (from_type, to_type) not in edge_type_to_from_to[edge_type]:
        edge_type_to_from_to[edge_type][(from_type, to_type)] = ([], [])
    from_idx = node_id_to_idx[from_type][from_id]
    to_idx = node_id_to_idx[to_type][to_id]
    edge_type_to_from_to[edge_type][(from_type, to_type)][0].append(from_idx)
    edge_type_to_from_to[edge_type][(from_type, to_type)][1].append(to_idx)

for edge_type, from_to_dict in edge_type_to_from_to.items():
    for (from_type, to_type), (from_idx, to_idx) in from_to_dict.items():
        edge_index = torch.tensor([from_idx, to_idx])
        data[from_type, edge_type, to_type].edge_index = edge_index

# Initialize random node features
for node_type in node_id_to_idx:
    num_nodes = data[node_type].num_nodes
    data[node_type].x = torch.randn(num_nodes, 128)

# Define GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        conv_dict = {}
        for edge_type in data.edge_types:
            from_type, _, to_type = edge_type
            conv_dict[edge_type] = SAGEConv(data[from_type].x.size(-1), hidden_channels)
        self.conv = HeteroConv(conv_dict, aggr='sum')

    def forward(self, data):
        x_dict = self.conv(data.x_dict, data.edge_index_dict)
        return x_dict

# Get embeddings
model = GNNModel(hidden_channels=128)
with torch.no_grad():
    output = model(data)

# Extract embeddings for class nodes only from TradePlatformService and TradeConfigService
class_nodes = [node['id'] for node in connected_nodes if node['type'] == 'class']
class_node_indices = {node['id']: idx for idx, node in enumerate(connected_nodes) if node['type'] == 'class'}
class_node_embs = output['class']

# Define the classes for TradePlatformService and TradeConfigService
trade_platform_classes = [
    'TradeAppJSF', 'PingServlet2Servlet', 'MarketSummaryDataBeanWS', 'MarketSummaryDataBean', 'TradeSLSBRemote',
    'PingBean', 'PingServlet2Jsp', 'TradeJDBCContextListener', 'PortfolioJSF', 'PingServlet2Session',
    'PingServlet2Session2CMROne2One', 'PingServlet2TwoPhase', 'OrdersAlertFilter', 'TradeJEEDirect',
    'PingServlet2Session2JDBC', 'RunStatsDataBean', 'AccountDataJSF'
]

trade_config_classes = [
    'TradeConfigJSF', 'PingServlet2Session2CMROne2Many', 'TradeAction', 'TradeDirectContextListener', 'TradeJPADirect',
    'PingServlet2Session2Entity', 'TradeServletAction', 'PingJDBCWrite', 'TradeConfigServlet', 'PingServlet2ServletRcv',
    'KeySequenceDirect', 'TradeSLSBBean', 'TradeJDBCDirect', 'TradeAppServlet', 'PingJDBCRead', 'FinancialUtils',
    'DirectSLSBBean'
]

# Combine relevant class nodes and their embeddings
relevant_classes = trade_platform_classes + trade_config_classes
relevant_indices = [class_node_indices[cls] for cls in relevant_classes if cls in class_node_indices]
relevant_embs = class_node_embs[relevant_indices]

# Function to get resources for a set of classes
def get_resources(classes, edges):
    resources = set()
    for edge in edges:
        if edge['type'] == 'accesses' and edge['from'] in classes:
            resources.add(edge['to'])
    return resources

# Get initial resources for each microservice
trade_platform_resources = get_resources(trade_platform_classes, graph['edges'])
trade_config_resources = get_resources(trade_config_classes, graph['edges'])

# Penalized K-means implementation
class PenalizedKMeans:
    def __init__(self, n_clusters, resource_penalty=1.0, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.resource_penalty = resource_penalty  # Penalty for resource overlap
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def fit(self, X, class_resources):
        n_samples = X.shape[0]
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        for iteration in range(self.max_iter):
            # Assign labels based on distance and resource penalty
            old_labels = self.labels_
            distances = euclidean_distances(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)

            # Calculate resource overlap penalty
            cluster_resources = {}
            for cluster_id in range(self.n_clusters):
                cluster_classes = [cls for i, cls in enumerate(relevant_classes) if self.labels_[i] == cluster_id]
                cluster_resources[cluster_id] = get_resources(cluster_classes, graph['edges'])

            # Adjust distances with resource penalty
            penalized_distances = distances.copy()
            for i in range(n_samples):
                current_cluster = self.labels_[i]
                for cluster_id in range(self.n_clusters):
                    if cluster_id != current_cluster:
                        overlap = len(cluster_resources[current_cluster] & cluster_resources[cluster_id])
                        penalized_distances[i, cluster_id] += self.resource_penalty * overlap

            self.labels_ = np.argmin(penalized_distances, axis=1)

            # Update centroids
            for cluster_id in range(self.n_clusters):
                cluster_points = X[self.labels_ == cluster_id]
                if len(cluster_points) > 0:
                    self.centroids[cluster_id] = np.mean(cluster_points, axis=0)

            # Check convergence
            if old_labels is not None and np.all(np.abs(old_labels - self.labels_) < self.tol):
                break

        return self

    def predict(self, X):
        return np.argmin(euclidean_distances(X, self.centroids), axis=1)

# Apply Penalized K-means
n_clusters = 5  # Target 2 clusters for TradePlatformService and TradeConfigService
class_resources = {cls: get_resources([cls], graph['edges']) for cls in relevant_classes}
penalized_kmeans = PenalizedKMeans(n_clusters=n_clusters, resource_penalty=1.0)
penalized_kmeans.fit(relevant_embs.numpy(), class_resources)

# Get final cluster assignments
new_clusters = penalized_kmeans.predict(relevant_embs.numpy())

# Group re-clustered classes into microservices
new_microservices = {}
for i, cls in enumerate(relevant_classes):
    cluster_id = new_clusters[i]
    if cluster_id not in new_microservices:
        new_microservices[cluster_id] = {'classes': [], 'resources': set()}
    new_microservices[cluster_id]['classes'].append(cls)
    new_microservices[cluster_id]['resources'].update(get_resources([cls], graph['edges']))

# Print re-clustered microservices
for cluster_id, content in new_microservices.items():
    classes = ', '.join(content['classes'])
    resources = ', '.join(content['resources']) if content['resources'] else 'None'
    print(f"Penalized KMeans-Microservice {cluster_id}:")
    print(f"  Classes: {classes}")
    print(f"  Resources: {resources}\n")

# Visualization (unchanged structure, but updated for new microservices)
G = nx.DiGraph()

# Add class nodes to their respective clusters in random circular groups
pos = {}
microservices = {k: {'classes': v['classes']} for k, v in new_microservices.items()}  # Update microservices for visualization
for cluster_id, functions in microservices.items():
    # Create a random position for the cluster center
    cluster_center = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
    # Create a circular layout for nodes within the cluster (small dots)
    n_nodes = len(functions['classes'])
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 0.1  # Small radius for dense grouping
    for i, func in enumerate(functions['classes']):
        angle = theta[i]
        x = cluster_center[0] + radius * np.cos(angle)
        y = cluster_center[1] + radius * np.sin(angle)
        pos[func] = (x, y)
        G.add_node(func, type='class', cluster=cluster_id)

# Add inter-cluster connections (directed edges between clusters, not nodes)
for cluster1_id, functions1 in microservices.items():
    for cluster2_id, functions2 in microservices.items():
        if cluster1_id < cluster2_id:  # Avoid duplicate edges
            for edge in graph['edges']:
                if edge['type'] == 'calls' and edge['from'] in functions1['classes'] and edge['to'] in functions2['classes']:
                    G.add_edge(f"Cluster{cluster1_id}", f"Cluster{cluster2_id}", type='inter-cluster')

# Add resource nodes as green small circles at the bottom
resource_nodes = [node['id'] for node in connected_nodes if node['type'] == 'resource']
for resource in resource_nodes:
    # Fixed position at the bottom, spread horizontally
    x = np.random.uniform(-1, 1)  # Random x position for spread
    y = -0.5  # Fixed at the bottom
    pos[resource] = (x, y)
    G.add_node(resource, type='resource')

# Add edges from clusters to resources
for cluster_id, functions in microservices.items():
    resources = new_microservices[cluster_id]['resources']
    for resource in resources:
        if f"Cluster{cluster_id}" in G and resource in G:
            G.add_edge(f"Cluster{cluster_id}", resource, type='accesses')

# Define node colors and shapes (small dots)
node_colors = []
node_shapes = []
node_sizes = []
for node in G.nodes():
    if node.startswith('Cluster'):
        node_colors.append('gray')  # Cluster nodes (not drawn, just for edges)
        node_shapes.append('^')     # Triangles (not drawn)
        node_sizes.append(0)        # Not drawn as nodes
    elif node in class_nodes:
        cluster_id = next(cid for cid, funcs in microservices.items() if node in funcs['classes'])
        node_colors.append(plt.cm.Set3(int(cluster_id) % len(plt.cm.Set3.colors)))  # Colors based on cluster
        node_shapes.append('o')     # Small circles for classes
        node_sizes.append(50)       # Small dot size
    else:  # resource
        node_colors.append('lightgreen')
        node_shapes.append('o')     # Small circles for resources
        node_sizes.append(50)       # Small dot size

# Draw the graph
plt.figure(figsize=(12, 8))  # Adjusted for random positioning

# Draw nodes as small dots (only classes and resources)
for shape in set(node_shapes):
    if shape == 'o':  # Only draw class and resource nodes (small dots)
        nodes = [n for n, s in zip(G.nodes(), node_shapes) if s == shape and n in pos]
        colors = [node_colors[i] for i, n in enumerate(G.nodes()) if node_shapes[i] == shape and n in pos]
        sizes = [node_sizes[i] for i, n in enumerate(G.nodes()) if node_shapes[i] == shape and n in pos]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_shape=shape, node_size=sizes)

# Draw edges with different styles (directed, visible, and thicker)
for edge in G.edges(data=True):
    source, target = edge[0], edge[1]
    if source in pos and target in pos:  # Ensure both nodes have positions
        print(f"Drawing edge {source} -> {target}: positions - {pos.get(source)}, {pos.get(target)}")
        if edge[2]['type'] == 'inter-cluster':
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], style='dashed', width=2.0, arrowsize=20, arrowstyle='->', edge_color='black', alpha=1.0)
        elif edge[2]['type'] == 'accesses':
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], style='dashed', width=2.0, arrowsize=20, arrowstyle='->', edge_color='black', alpha=1.0)

# Remove individual node labels (no nx.draw_networkx_labels)
# Add cluster labels (for cluster centers, positioned at cluster center)
for cluster_id, functions in microservices.items():
    if functions['classes']:
        # Use the average position of nodes in the cluster as the label position
        positions = [pos[func] for func in functions['classes'] if func in pos]
        if positions:
            avg_pos = (sum(p[0] for p in positions) / len(positions), sum(p[1] for p in positions) / len(positions))
            plt.text(avg_pos[0], avg_pos[1] + 0.1, f"Cluster {cluster_id}", 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

# Add a legend
plt.scatter([], [], c='lightblue', label='Class', s=50, marker='o')
plt.scatter([], [], c='lightgreen', label='Resource', s=50, marker='o')
plt.plot([], [], 'k--', label='Inter-Cluster/Accesses Edge')
plt.legend(scatterpoints=1, loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.8)

# Finalize plot
plt.title("Penalized K-means Clusters with Random Positions, Inter-Cluster Connections, and Resource Dependencies", pad=20, fontsize=14)
plt.axis('off')
plt.tight_layout()
# plt.show()