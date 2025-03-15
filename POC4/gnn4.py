#KMeans on Class Node Embeddings by GNN, with enhanced egdes between classes with common resources
import json
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

# Function to get resources for a set of classes or a single class
def get_resources(classes_or_class, edges):
    resources = set()
    if isinstance(classes_or_class, str):  # Single class
        class_id = classes_or_class
        for edge in edges:
            if edge['type'] == 'accesses' and edge['from'] == class_id:
                resources.add(edge['to'])
    else:  # List of classes
        for class_id in classes_or_class:
            for edge in edges:
                if edge['type'] == 'accesses' and edge['from'] == class_id:
                    resources.add(edge['to'])
    return resources

# Create node ID to index mapping for connected nodes
node_id_to_idx = {}
for node in connected_nodes:
    node_type = node['type']
    if node_type not in node_id_to_idx:
        node_id_to_idx[node_type] = {}
    node_id_to_idx[node_type][node['id']] = len(node_id_to_idx[node_type])

# Enhance edges: add edges between all classes that access the same resources
enhanced_edges = connected_edges.copy()
class_nodes = [node['id'] for node in connected_nodes if node['type'] == 'class']
resource_access_map = {cls: get_resources(cls, connected_edges) for cls in class_nodes}

for cls1 in class_nodes:
    for cls2 in class_nodes:
        if cls1 < cls2:  # Avoid duplicate edges
            resources1 = resource_access_map[cls1]
            resources2 = resource_access_map[cls2]
            if resources1 & resources2:  # If classes share any resources
                enhanced_edges.append({
                    'from': cls1,
                    'to': cls2,
                    'type': 'resource_shared'
                })

# Create HeteroData object with enhanced edges
data = HeteroData()
for node_type in ['class', 'resource']:  # Only include relevant node types
    data[node_type].num_nodes = len([n for n in connected_nodes if n['type'] == node_type])

# Add original and enhanced edges
edge_type_to_from_to = {}
for edge in enhanced_edges:
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




from matplotlib.lines import Line2D
G = nx.DiGraph()  # Directed graph for visualization
# Dictionaries for visualization
node_labels = {}
node_colors = {}
edge_colors = []
edge_labels = {}

# Add nodes with labels and colors
for node in connected_nodes:
    G.add_node(node['id'])
    node_labels[node['id']] = node['id']
    node_colors[node['id']] = "skyblue" if node['type'] == "class" else "lightgreen"

# Add edges with labels and colors
for edge in enhanced_edges:
    from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
    G.add_edge(from_id, to_id, label=edge_type)
    edge_labels[(from_id, to_id)] = edge_type
    edge_colors.append("green" if edge_type == "calls" else ("blue" if edge_type == "accesses" else "red"))

# Generate improved layout
pos = nx.spring_layout(G, seed=42)  # Ensures consistency in layout

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=[node_colors[n] for n in G.nodes()],
        node_size=2000, font_size=10, edge_color=edge_colors, linewidths=1.5, edgecolors="black")

# Draw edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=9)

# Create a legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Class Node', markersize=10, markerfacecolor='skyblue'),
    Line2D([0], [0], marker='o', color='w', label='Resource Node', markersize=10, markerfacecolor='lightgreen'),
    Line2D([0], [0], color='green', lw=2, label='Calls Edge'),
    Line2D([0], [0], color='blue', lw=2, label='Accesses Edge'),
    Line2D([0], [0], color='red', lw=2, label='Resource_Shared Edge')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True)

plt.title("Improved Heterogeneous Graph Visualization")
plt.show()

# Initialize random node features
for node_type in ['class', 'resource']:
    num_nodes = data[node_type].num_nodes
    data[node_type].x = torch.randn(num_nodes, 128)

# Define GNN model with enhanced edges
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

# Extract embeddings for all class nodes
class_nodes = [node['id'] for node in connected_nodes if node['type'] == 'class']
class_node_indices = {node['id']: idx for idx, node in enumerate(connected_nodes) if node['type'] == 'class'}
class_node_embs = output['class']
relevant_indices = [class_node_indices[cls] for cls in class_nodes]
relevant_embs = class_node_embs[relevant_indices]

# Apply K-means clustering to all class nodes
n_clusters = 6  # Adjust based on the total number of expected microservices (e.g., from your KMeans output)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
new_clusters = kmeans.fit_predict(relevant_embs.numpy())

# Group all re-clustered classes into microservices
new_microservices = {}
for i, cls in enumerate(class_nodes):
    cluster_id = new_clusters[i]
    if cluster_id not in new_microservices:
        new_microservices[cluster_id] = {'classes': [], 'resources': set()}
    new_microservices[cluster_id]['classes'].append(cls)
    # Add resources based on "accesses" edges
    new_microservices[cluster_id]['resources'].update(get_resources([cls], graph['edges']))

# Print re-clustered microservices
for cluster_id, content in new_microservices.items():
    classes = ', '.join(content['classes'])
    resources = ', '.join(content['resources']) if content['resources'] else 'None'
    print(f"Resource-Based KMeans-Microservice {cluster_id}:")
    print(f"Classes: {classes}")
    print(f"Resources: {resources}\n")

##################################################################################################

import networkx as nx
from community import community_louvain

# Create a NetworkX graph from the enhanced edges for Louvain clustering
louvain_graph = nx.Graph()

# Add class nodes
for cls in class_nodes:
    louvain_graph.add_node(cls, type='class')

# Add original and enhanced edges (calls and resource_shared) between classes
for edge in enhanced_edges:
    if edge['type'] in ['calls', 'resource_shared'] and edge['from'] in class_nodes and edge['to'] in class_nodes:
        louvain_graph.add_edge(edge['from'], edge['to'])

# Apply Louvain clustering
louvain_partition = community_louvain.best_partition(louvain_graph)

# Group Louvain-clustered classes into microservices
louvain_microservices = {}
for node, cluster_id in louvain_partition.items():
    if cluster_id not in louvain_microservices:
        louvain_microservices[cluster_id] = {'classes': [], 'resources': set()}
    louvain_microservices[cluster_id]['classes'].append(node)
    # Add resources based on "accesses" edges
    louvain_microservices[cluster_id]['resources'].update(get_resources([node], graph['edges']))

# Print Louvain-clustered microservices
# print("\nLouvain-Clustered Microservices:")
# for cluster_id, content in louvain_microservices.items():
#     classes = ', '.join(content['classes'])
#     resources = ', '.join(content['resources']) if content['resources'] else 'None'
#     print(f"Louvain-Microservice {cluster_id}:")
#     print(f"Classes: {classes}")
#     print(f"Resources: {resources}\n")

# # Update visualization for Louvain clusters (optional, if you want to visualize both)
# # Modify the microservices dictionary for visualization to use Louvain clusters
# louvain_microservices_vis = {k: {'classes': v['classes']} for k, v in louvain_microservices.items()}

# # Replace the K-means microservices in the visualization with Louvain microservices
# microservices = louvain_microservices_vis

# (The rest of the visualization code remains the same, starting from adding class nodes to their clusters)

# # Visualization
# G = nx.DiGraph()

# # Add class nodes to their respective clusters in random circular groups
# pos = {}
# microservices = {k: {'classes': v['classes']} for k, v in new_microservices.items()}  # Update microservices for visualization
# for cluster_id, functions in microservices.items():
#     # Create a random position for the cluster center
#     cluster_center = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
#     # Create a circular layout for nodes within the cluster (small dots)
#     n_nodes = len(functions['classes'])
#     theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
#     radius = 0.1  # Small radius for dense grouping
#     for i, func in enumerate(functions['classes']):
#         angle = theta[i]
#         x = cluster_center[0] + radius * np.cos(angle)
#         y = cluster_center[1] + radius * np.sin(angle)
#         pos[func] = (x, y)
#         G.add_node(func, type='class', cluster=cluster_id)

# # Add inter-cluster connections (directed edges between clusters, not nodes)
# for cluster1_id, functions1 in microservices.items():
#     for cluster2_id, functions2 in microservices.items():
#         if cluster1_id < cluster2_id:  # Avoid duplicate edges
#             for edge in graph['edges']:
#                 if edge['type'] == 'calls' and edge['from'] in functions1['classes'] and edge['to'] in functions2['classes']:
#                     G.add_edge(f"Cluster{cluster1_id}", f"Cluster{cluster2_id}", type='inter-cluster')

# # Add resource nodes as green small circles at the bottom
# resource_nodes = [node['id'] for node in connected_nodes if node['type'] == 'resource']
# for resource in resource_nodes:
#     # Fixed position at the bottom, spread horizontally
#     x = np.random.uniform(-1, 1)  # Random x position for spread
#     y = -0.5  # Fixed at the bottom
#     pos[resource] = (x, y)
#     G.add_node(resource, type='resource')

# # Add edges from clusters to resources
# for cluster_id, functions in microservices.items():
#     resources = new_microservices[cluster_id]['resources']
#     for resource in resources:
#         if f"Cluster{cluster_id}" in G and resource in G:
#             G.add_edge(f"Cluster{cluster_id}", resource, type='accesses')

# # Define node colors and shapes (small dots)
# node_colors = []
# node_shapes = []
# node_sizes = []
# for node in G.nodes():
#     if node.startswith('Cluster'):
#         node_colors.append('gray')  # Cluster nodes (not drawn, just for edges)
#         node_shapes.append('^')     # Triangles (not drawn)
#         node_sizes.append(0)        # Not drawn as nodes
#     elif node in class_nodes:
#         cluster_id = next(cid for cid, funcs in microservices.items() if node in funcs['classes'])
#         node_colors.append(plt.cm.Set3(int(cluster_id) % len(plt.cm.Set3.colors)))  # Colors based on cluster
#         node_shapes.append('o')     # Small circles for classes
#         node_sizes.append(50)       # Small dot size
#     else:  # resource
#         node_colors.append('lightgreen')
#         node_shapes.append('o')     # Small circles for resources
#         node_sizes.append(50)       # Small dot size

# # Draw the graph
# plt.figure(figsize=(12, 8))  # Adjusted for random positioning

# # Draw nodes as small dots (only classes and resources)
# for shape in set(node_shapes):
#     if shape == 'o':  # Only draw class and resource nodes (small dots)
#         nodes = [n for n, s in zip(G.nodes(), node_shapes) if s == shape and n in pos]
#         colors = [node_colors[i] for i, n in enumerate(G.nodes()) if node_shapes[i] == shape and n in pos]
#         sizes = [node_sizes[i] for i, n in enumerate(G.nodes()) if node_shapes[i] == shape and n in pos]
#         nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_shape=shape, node_size=sizes)

# # Draw edges with different styles (directed, visible, and thicker)
# for edge in G.edges(data=True):
#     source, target = edge[0], edge[1]
#     if source in pos and target in pos:  # Ensure both nodes have positions
#         print(f"Drawing edge {source} -> {target}: positions - {pos.get(source)}, {pos.get(target)}")
#         if edge[2]['type'] == 'inter-cluster':
#             nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], style='dashed', width=2.0, arrowsize=20, arrowstyle='->', edge_color='black', alpha=1.0)
#         elif edge[2]['type'] == 'accesses':
#             nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], style='dashed', width=2.0, arrowsize=20, arrowstyle='->', edge_color='black', alpha=1.0)

# # Remove individual node labels (no nx.draw_networkx_labels)
# # Add cluster labels (for cluster centers, positioned at cluster center)
# for cluster_id, functions in microservices.items():
#     if functions['classes']:
#         # Use the average position of nodes in the cluster as the label position
#         positions = [pos[func] for func in functions['classes'] if func in pos]
#         if positions:
#             avg_pos = (sum(p[0] for p in positions) / len(positions), sum(p[1] for p in positions) / len(positions))
#             plt.text(avg_pos[0], avg_pos[1] + 0.1, f"Cluster {cluster_id}", 
#                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2), 
#                      ha='center', va='center', fontsize=12, fontweight='bold')

# # Add a legend
# plt.scatter([], [], c='lightblue', label='Class', s=50, marker='o')
# plt.scatter([], [], c='lightgreen', label='Resource', s=50, marker='o')
# plt.plot([], [], 'k--', label='Inter-Cluster/Accesses Edge')
# plt.legend(scatterpoints=1, loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.8)

# # Finalize plot
# plt.title("Generic Resource-Based K-means Clusters with Random Positions, Inter-Cluster Connections, and Resource Dependencies", pad=20, fontsize=14)
# plt.axis('off')
# plt.tight_layout()
# # plt.show()




#################### Metrics #############################

import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score

# ================== HARD-CODED MICROSERVICES ==================
microservices = {
    "TradeSetupService": [
        "HoldingData", "TradeAppJSF", "TradeBuildDB", "TradeDirectContextListener",
        "TradeJPAContextListener", "PingServlet2Jsp"
    ],
    "TradePlatformService": [
        "PingServlet2Session2JDBCCollection", "PingServlet2Servlet", "MarketSummaryDataBeanWS",
        "PingServlet2Session2CMROne2Many", "TradeAction", "TradeJPADirect", "MarketSummaryDataBean",
        "TradeSLSBRemote", "PingServlet2Session2Entity", "TradeServletAction", "TradeJDBCContextListener",
        "TradeConfigServlet", "TradeServices", "PortfolioJSF", "PingServlet2ServletRcv",
        "KeySequenceDirect", "TradeSLSBBean", "TradeJDBCDirect", "PingServlet2Session2CMROne2One",
        "PingServlet2TwoPhase", "TradeAppServlet", "TradeJEEDirect", "RunStatsDataBean", "MarketSummaryJSF"
    ],
    "OrderConfigService": [
        "TradeConfigJSF", "PingJDBCRead", "OrderData", "AccountDataJSF", "OrderDataJSF"
    ],
    "QuoteDataService": [
        "QuoteDataJSF", "QuoteData"
    ],
    "TradeUtilityService": [
        "PingBean", "PingServlet2Session", "FinancialUtils", "DirectSLSBBean",
        "PingServlet2Session2JDBC", "PingServlet2Session2EntityCollection"
    ],
    "TradeDBOrderService": [
        "PingJDBCWrite", "OrdersAlertFilter", "TradeDBServices"
    ]
}

# ================== HARD-CODED RESOURCE ACCESS ==================
resource_access = {
    "TradePlatformService": {"order", "holding", "quote", "account", "accountprofile"},
    "TradeSetupService": set(),
    "OrderConfigService": set(),
    "QuoteDataService": set(),
    "TradeUtilityService": set(),
    "TradeDBOrderService": set()
}


# ================== GRAPH-BASED METRICS ==================
# **Step 5: Compute Silhouette Score**
# Assume `class_node_embs` is the embedding matrix (hardcoded or loaded separately)
label_mapping = {cls: cluster_id for cluster_id, classes in microservices.items() for cls in classes}
# Convert PyTorch tensor to NumPy array if needed
import torch

# Convert PyTorch tensor to NumPy array (if it's not already)
if isinstance(class_node_embs, torch.Tensor):
    class_node_embs = class_node_embs.numpy()

# Ensure we have a dictionary mapping node names to their embeddings
node_to_embedding = {node: emb for node, emb in zip(class_nodes, class_node_embs)}

# Extract embeddings in the correct order
emb_array = np.array([node_to_embedding[node] for node in G.nodes if node in node_to_embedding])

# Ensure labels match the same order
labels = np.array([label_mapping[cls] for cls in G.nodes if cls in label_mapping])

if len(set(labels)) > 1:  # Silhouette requires at least 2 clusters
    silhouette = silhouette_score(emb_array, labels)
else:
    silhouette = -1  # Undefined silhouette for a single cluster

print(f"Silhouette Score: {silhouette:.4f}")

# **Step 6: Compute Modularity Score**
# Convert clusters into community format for modularity calculation
communities = [set(classes) for classes in microservices.values()]
class_nodes_updated = {node for node in G.nodes if node in label_mapping}  # Only keep class nodes

# Step 2: Filter communities to contain only class nodes
filtered_communities = []
for cluster in communities:
    filtered_cluster = {node for node in cluster if node in class_nodes_updated}
    if filtered_cluster:  # Only add non-empty clusters
        filtered_communities.append(filtered_cluster)

# Step 3: Compute modularity with filtered communities
modularity_score = nx.algorithms.community.quality.modularity(G.subgraph(class_nodes_updated), filtered_communities)

print(f"Modularity Score: {modularity_score:.4f}")

# # **Step 7: Compute Service Coupling (Shared Resources)**
# coupling = sum(len(resource_access[c1].intersection(resource_access[c2])) 
#                for c1 in resource_access for c2 in resource_access if c1 != c2)

# print(f"Service Coupling Score: {coupling}")

# Compute Service Coupling (Shared Resources)
inter_service_resource_sharing = 0
total_resource_accesses = sum(len(res) for res in resource_access.values())  # Total accesses

for c1 in resource_access:
    for c2 in resource_access:
        if c1 != c2:
            inter_service_resource_sharing += len(resource_access[c1].intersection(resource_access[c2]))

# Normalize by total resource accesses
service_coupling = inter_service_resource_sharing / total_resource_accesses if total_resource_accesses > 0 else 0
print(f"Service Coupling Score: {service_coupling:.4f}")


# **Step 8: Compute Service Cohesion (Internal Dependencies)**
# cohesion = 0
# for cluster_id, classes in microservices.items():
#     internal_edges = sum(1 for c1 in classes for c2 in classes if G.has_edge(c1, c2))
#     cohesion += internal_edges

# print(f"Service Cohesion Score: {cohesion}")

# Compute Service Cohesion (Internal Dependencies)
cohesion_scores = []
for cluster_id, classes in microservices.items():
    internal_edges = sum(1 for c1 in classes for c2 in classes if G.has_edge(c1, c2))  
    total_service_calls = sum(1 for c1 in classes for c2 in G.neighbors(c1))  # All calls

    # Normalize cohesion within the service
    cohesion_score = internal_edges / total_service_calls if total_service_calls > 0 else 0
    cohesion_scores.append(cohesion_score)

# Take the average cohesion across all microservices
service_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0
print(f"Service Cohesion Score: {service_cohesion:.4f}")
