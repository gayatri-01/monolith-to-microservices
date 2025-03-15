#KMeans on Class Node Embeddings by GNN for DayTrader App
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

# Filter edges to only include connected nodes
connected_edges = [edge for edge in graph['edges'] if edge['from'] in connected_node_ids and edge['to'] in connected_node_ids]

# Create node ID to index mapping
node_id_to_idx = {}
for node in connected_nodes:
    node_type = node['type']
    if node_type not in node_id_to_idx:
        node_id_to_idx[node_type] = {}
    node_id_to_idx[node_type][node['id']] = len(node_id_to_idx[node_type])


# Create HeteroData object
data = HeteroData()
for node_type in node_id_to_idx:
    data[node_type].num_nodes = len(node_id_to_idx[node_type])

# Add edges
edge_type_to_from_to = {}
for edge in connected_edges:
    from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
    from_type = next(node['type'] for node in graph['nodes'] if node['id'] == from_id)
    to_type = next(node['type'] for node in graph['nodes'] if node['id'] == to_id)
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

################################################################################


# G = nx.DiGraph()  # Directed graph for visualization
# node_labels = {}  # Store labels for visualization
# node_colors = {}  # Store node colors based on type
# edge_colors = []  # Store edge colors
# edge_labels = {}  # Store edge labels

# for node in connected_nodes:
#     G.add_node(node['id'])
#     node_type = node['type'] 
#     node_labels[node['id']] = node['id']
#     # Assign colors based on node type
#     if node_type == "class":
#         node_colors[node['id']] = "skyblue"
#     elif node_type == "resource":
#         node_colors[node['id']] = "lightgreen"


# for edge in connected_edges:
#     from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
#     G.add_edge(from_id, to_id, label=edge_type)
    
#     edge_labels[(from_id, to_id)] = edge_type  # Label edges
#     if edge_type == "calls":
#         edge_colors.append("red")
#     elif edge_type == "accesses":
#         edge_colors.append("blue")

# # Generate layout (better for disconnected graphs)
# pos = nx.kamada_kawai_layout(G)

# # Ensure all nodes have a position
# for node in G.nodes():
#     if node not in pos:
#         pos[node] = (0, 0)  # Assign default position

# # Draw the graph with different node colors
# plt.figure(figsize=(12, 8))

# # Get node colors in the same order as the node list
# node_color_list = [node_colors[n] for n in G.nodes()]
# nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_color_list,
#         node_size=2000, font_size=10, edge_color=edge_colors)

# # Draw edge labels
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black")

# # Add a legend
# from matplotlib.lines import Line2D

# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label='Class Node', markersize=10, markerfacecolor='skyblue'),
#     Line2D([0], [0], marker='o', color='w', label='Resource Node', markersize=10, markerfacecolor='lightgreen'),
#     Line2D([0], [0], color='red', lw=2, label='Calls Edge'),
#     Line2D([0], [0], color='blue', lw=2, label='Accesses Edge')
# ]
# plt.legend(handles=legend_elements, loc='upper right')

# plt.title("Heterogeneous Graph Visualization")
# plt.show()

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
for edge in connected_edges:
    from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
    G.add_edge(from_id, to_id, label=edge_type)
    edge_labels[(from_id, to_id)] = edge_type
    edge_colors.append("red" if edge_type == "calls" else "blue")

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
    Line2D([0], [0], color='red', lw=2, label='Calls Edge'),
    Line2D([0], [0], color='blue', lw=2, label='Accesses Edge')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True)

plt.title("Improved Heterogeneous Graph Visualization")
plt.show()

################################################################################

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
# class_nodes = [node['id'] for node in graph['nodes'] if node['type'] == 'class']
# class_node_indices = [node_id_to_idx['class'][nid] for nid in class_nodes]
# class_node_embs = output['class'][class_node_indices]

class_nodes = [node['id'] for node in connected_nodes if node['type'] == 'class']
resource_nodes = [node['id'] for node in connected_nodes if node['type'] == 'resource']
class_node_indices = [node_id_to_idx['class'][nid] for nid in class_nodes]
resource_node_indices = [node_id_to_idx['resource'][nid] for nid in resource_nodes]
class_node_embs = output['class'][class_node_indices]
resource_node_embs = output['resource'][resource_node_indices]

# Combine embeddings for all nodes
# all_nodes = class_nodes + resource_nodes
# all_embs = torch.cat([class_node_embs, resource_node_embs], dim=0)

def generate_service_name(functions):
    prompt = (
        "You are an expert in software architecture. Given the following Class names of the DayTrader Application, "
        "suggest a meaningful microservice name:\n\n"
        f"Classes: {', '.join(functions)}\n\n"
        "The name should be concise, descriptive, and follow naming conventions like 'UserService', "
        "'OrderProcessingService', etc. It should end with 'Service'\n\n"
        "Output only the name with no explanation."
        "IMPORTANT - DO NOT use the same name for multiple services. \n\n"
        "IMPORTANT - ONLY ONE WORD RESPONSE!!!!!"
    )

    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()


############## [1] KMeans Clustering with Both Classes & Resources ##############


# Apply K-means clustering
k = 7  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(class_node_embs.numpy())

microservices = {}
for func, cluster_id in zip(class_nodes, clusters):
    if cluster_id not in microservices:
        microservices[cluster_id] = []
    microservices[cluster_id].append(func)



# for cluster_id, functions in microservices.items():
    # print(f"KMeans-Microservice {cluster_id}: {functions}")
    # print(f"Microservice Name: {generate_service_name(functions)}\n")

# Group nodes by clusters
# cluster_dict = {}
# for i, node in enumerate(all_nodes):
#     cluster_id = clusters[i]
#     if cluster_id not in cluster_dict:
#         cluster_dict[cluster_id] = {'classes': [], 'resources': []}
#     if node in class_nodes:
#         cluster_dict[cluster_id]['classes'].append(node)
#     elif node in resource_nodes:
#         cluster_dict[cluster_id]['resources'].append(node)

# # Print clusters with classes and their associated resources
# for cluster_id, content in cluster_dict.items():
#     classes = ', '.join(content['classes']) if content['classes'] else 'None'
#     resources = ', '.join(content['resources']) if content['resources'] else 'None'
#     print(f"Cluster {cluster_id}:")
#     print(f"Classes: {classes}")
#     print(f"Resources: {resources}")
    # print(f"Microservice Name: {generate_service_name(classes)}\n")

############### Visualise the clusters ####################
# Print microservices with their resource dependencies (unchanged)
for cluster_id, functions in microservices.items():
    resources = set()
    for edge in graph['edges']:
        if edge['type'] == 'accesses' and edge['from'] in functions:
            resources.add(edge['to'])
    print(f"KMeans-Microservice {cluster_id}:")
    print(f"Classes: {functions}")
    print(f"Resources: {list(resources) if resources else 'None'}\n")




# Create a NetworkX directed graph for visualization of clusters
G = nx.DiGraph()

# Add class nodes to their respective clusters in random circular groups
pos = {}  # Random positions for clusters
for cluster_id, functions in microservices.items():
    # Create a random position for the cluster center
    cluster_center = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
    # Create a circular layout for nodes within the cluster (small dots)
    n_nodes = len(functions)
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 0.1  # Small radius for dense grouping
    for i, func in enumerate(functions):
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
                if edge['type'] == 'calls' and edge['from'] in functions1 and edge['to'] in functions2:
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
    resources = set()
    for edge in graph['edges']:
        if edge['type'] == 'accesses' and edge['from'] in functions:
            resources.add(edge['to'])
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
        cluster_id = next(cid for cid, funcs in microservices.items() if node in funcs)
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
# Draw edges with different styles (directed, visible, and thicker)
for edge in G.edges(data=True):
    source, target = edge[0], edge[1]
    if source in pos and target in pos:  # Ensure both nodes have positions
        if edge[2]['type'] == 'inter-cluster':
            print(f"Drawing edge {source} -> {target}: positions - {pos.get(source)}, {pos.get(target)}")
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], style='dashed', width=2.0, arrowsize=20, arrowstyle='->', edge_color='black', alpha=1.0)
        elif edge[2]['type'] == 'accesses':
            print(f"Drawing edge {source} -> {target}: positions - {pos.get(source)}, {pos.get(target)}")
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], style='dashed', width=2.0, arrowsize=20, arrowstyle='->', edge_color='black', alpha=1.0)

# Remove individual node labels (no nx.draw_networkx_labels)
# Add cluster labels (for cluster centers, positioned at cluster center)
for cluster_id, functions in microservices.items():
    if functions:
        # Use the average position of nodes in the cluster as the label position
        positions = [pos[func] for func in functions if func in pos]
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
plt.title("K-means Clusters with Random Positions, Inter-Cluster Connections, and Resource Dependencies", pad=20, fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

# print("Edges to draw:", [(edge[0], edge[1], edge[2]['type']) for edge in G.edges(data=True)])
###########################################################

# # Create a NetworkX graph for visualization
# G = nx.Graph()

# # Add nodes and edges from clusters
# for cluster_id, content in cluster_dict.items():
#     # Add class nodes
#     for class_node in content['classes']:
#         G.add_node(class_node, type='class', cluster=cluster_id)
#     # Add resource nodes
#     for resource_node in content['resources']:
#         G.add_node(resource_node, type='resource', cluster=cluster_id)

# # Add edges from the original graph.json
# for edge in graph['edges']:
#     from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
#     # Only add edges if both nodes are in the clustered graph
#     if G.has_node(from_id) and G.has_node(to_id):
#         G.add_edge(from_id, to_id, type=edge_type)

# # Define node colors and shapes
# node_colors = []
# node_shapes = []
# for node in G.nodes(data=True):
#     if node[1]['type'] == 'class':
#         node_colors.append('lightblue')  # Classes in light blue
#         node_shapes.append('o')          # Circle shape for classes
#     else:  # resource
#         node_colors.append('lightgreen') # Resources in light green
#         node_shapes.append('s')          # Square shape for resources

# # Position nodes using a spring layout, seeded for reproducibility
# pos = nx.spring_layout(G, seed=42)

# # Draw the graph
# plt.figure(figsize=(12, 8))

# # Draw nodes with different shapes (requires drawing separately for each shape)
# for shape in set(node_shapes):
#     nodes = [n for n, s in zip(G.nodes(), node_shapes) if s == shape]
#     colors = [node_colors[i] for i, n in enumerate(G.nodes()) if node_shapes[i] == shape]
#     nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_shape=shape, node_size=500)

# # Draw edges with different styles
# for edge in G.edges(data=True):
#     if edge[2]['type'] == 'calls':
#         nx.draw_networkx_edges(G, pos, edgelist=[edge], style='solid', width=1)
#     elif edge[2]['type'] == 'accesses':
#         nx.draw_networkx_edges(G, pos, edgelist=[edge], style='dashed', width=1)

# # Draw labels
# nx.draw_networkx_labels(G, pos, font_size=8)

# # Add cluster annotations
# for cluster_id, content in cluster_dict.items():
#     if content['classes'] or content['resources']:
#         # Approximate cluster center by averaging positions of its nodes
#         cluster_nodes = content['classes'] + content['resources']
#         if cluster_nodes:
#             avg_pos = [sum(pos[node][dim] for node in cluster_nodes) / len(cluster_nodes) for dim in range(2)]
#             plt.text(avg_pos[0], avg_pos[1] + 0.05, f"Cluster {cluster_id}", 
#                      bbox=dict(facecolor='white', alpha=0.7), ha='center', va='center')

# # Add a legend
# plt.scatter([], [], c='lightblue', label='Class', s=100)
# plt.scatter([], [], c='lightgreen', label='Resource', s=100)
# plt.plot([], [], 'k-', label='Calls Edge')
# plt.plot([], [], 'k--', label='Accesses Edge')
# plt.legend(scatterpoints=1, loc='upper right')

# # Finalize plot
# plt.title("K-means Clustering of Classes and Resources")
# plt.axis('off')
# #plt.show()


################# [2] Louvain Clustering on Plain Graph Representation without Resources or Embeddings ######################
# Create a NetworkX graph for Louvain clustering (only for class nodes)
G = nx.Graph()
for edge in connected_edges:
    if edge['type'] == 'calls':  # Only consider method calls between classes
        from_id, to_id = edge['from'], edge['to']
        if from_id in class_nodes and to_id in class_nodes:  # Ensure both are class nodes
            G.add_edge(from_id, to_id)

# Apply Louvain clustering
partition = community_louvain.best_partition(G)

# Group classes by clusters
clusters = {}
for node, cluster_id in partition.items():
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(node)

# # Print clusters
# for cluster_id, class_list in clusters.items():
#     print(f"Louvain-Cluster {cluster_id}: {', '.join(class_list)}")


# for cluster_id, functions in clusters.items():
#     resources = set()
#     for edge in graph['edges']:
#         if edge['type'] == 'accesses' and edge['from'] in functions:
#             resources.add(edge['to'])
#     print(f"Louvain-Microservice {cluster_id}:")
#     print(f"Classes: {functions}")
#     print(f"Resources: {list(resources) if resources else 'None'}\n")

############# [3] Louvain Clustering with Similarity Matrix [NOT GOOD] ######################


# Compute similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(class_node_embs.numpy())

# Create a NetworkX graph based on embedding similarity
G = nx.Graph()
for i, node_i in enumerate(class_nodes):
    G.add_node(node_i)
    for j, node_j in enumerate(class_nodes):
        if i < j:  # Avoid self-loops and duplicates
            similarity = similarity_matrix[i][j]
            if similarity > 0.3:  # Threshold for adding an edge (adjustable)
                G.add_edge(node_i, node_j, weight=similarity)

# Apply Louvain clustering
partition = community_louvain.best_partition(G, weight='weight')

# Group classes by clusters
clusters = {}
for node, cluster_id in partition.items():
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(node)

# Print clusters
# for cluster_id, class_list in clusters.items():
#     print(f"Louvain-Cluster {cluster_id}: {', '.join(class_list)}")

##################### [4] Louvain Clustering without Similarity Matrix  [NOT GOOD] ########################

class_nodes = [node['id'] for node in graph['nodes'] if node['type'] == 'class']
resource_nodes = [node['id'] for node in graph['nodes'] if node['type'] == 'resource']

# Create a NetworkX graph including both class and resource nodes
G = nx.Graph()
# Add all nodes
for node in class_nodes + resource_nodes:
    G.add_node(node)
# Add edges with weights
for edge in graph['edges']:
    from_id, to_id, edge_type = edge['from'], edge['to'], edge['type']
    if edge_type == 'calls' and from_id in class_nodes and to_id in class_nodes:
        G.add_edge(from_id, to_id, weight=1.0)  # Weight for class-to-class calls
    elif edge_type == 'accesses' and from_id in class_nodes and to_id in resource_nodes:
        G.add_edge(from_id, to_id, weight=0.5)  # Weight for class-to-resource accesses

# Apply Louvain clustering
partition = community_louvain.best_partition(G, weight='weight')

# Group nodes by clusters
clusters = {}
for node, cluster_id in partition.items():
    if cluster_id not in clusters:
        clusters[cluster_id] = {'classes': [], 'resources': []}
    if node in class_nodes:
        clusters[cluster_id]['classes'].append(node)
    elif node in resource_nodes:
        clusters[cluster_id]['resources'].append(node)

# Print clusters with classes and their associated resources
# for cluster_id, content in clusters.items():
#     classes = ', '.join(content['classes']) if content['classes'] else 'None'
#     resources = ', '.join(content['resources']) if content['resources'] else 'None'
#     print(f"Louvain-Cluster-Updated {cluster_id}:")
#     print(f"  Classes: {classes}")
#     print(f"  Resources: {resources}")





#################### Metrics #############################

import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score

# ================== HARD-CODED MICROSERVICES ==================
microservices = {
    1: ['HoldingData', 'PingServlet2Session2JDBCCollection', 'TradeJPAContextListener', 'TradeServices',
        'PingServlet2Session2EntityCollection', 'MarketSummaryJSF'],
    0: ['TradeAppJSF', 'PingServlet2Servlet', 'MarketSummaryDataBeanWS', 'MarketSummaryDataBean',
        'TradeSLSBRemote', 'PingBean', 'PingServlet2Jsp', 'TradeJDBCContextListener', 'PortfolioJSF',
        'PingServlet2Session', 'PingServlet2Session2CMROne2One', 'PingServlet2TwoPhase', 'OrdersAlertFilter',
        'TradeJEEDirect', 'PingServlet2Session2JDBC', 'RunStatsDataBean', 'AccountDataJSF'],
    3: ['TradeConfigJSF', 'PingServlet2Session2CMROne2Many', 'TradeAction', 'TradeDirectContextListener',
        'TradeJPADirect', 'PingServlet2Session2Entity', 'TradeServletAction', 'PingJDBCWrite',
        'TradeConfigServlet', 'PingServlet2ServletRcv', 'KeySequenceDirect', 'TradeSLSBBean',
        'TradeJDBCDirect', 'TradeAppServlet', 'PingJDBCRead', 'FinancialUtils', 'DirectSLSBBean'],
    4: ['TradeBuildDB', 'TradeDBServices'],
    5: ['QuoteDataJSF', 'QuoteData'],
    2: ['OrderData', 'OrderDataJSF']
}

# ================== HARD-CODED RESOURCE ACCESS ==================
resource_access = {
    0: {'holding', 'quote', 'account', 'order', 'accountprofile'},
    3: {'holding', 'quote', 'accountprofile', 'account', 'order'},
    4: set(),  # No shared resources
    5: set(),
    2: set(),
    1: set()
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

# **Step 7: Compute Service Coupling (Shared Resources)**
# coupling = sum(len(resource_access[c1].intersection(resource_access[c2])) 
#                for c1 in resource_access for c2 in resource_access if c1 != c2)

# print(f"Service Coupling Score: {coupling}")

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




#########################################################




## TODO
# 1. use LLM to predict Microservice Names for KNN Approach
# 2. Visualise the clusters (After naming & before naming)
# 3. Make the Heterogeneous Graph visualization better and interactive