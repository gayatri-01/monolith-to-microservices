import yaml
import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.metrics import pairwise_distances

# Step 1 & 2: Load OpenAPI spec and extract operation names
def load_operation_names(file_path):
    """Load OpenAPI YAML file and extract operation names."""
    with open("twitter.yaml", "r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)
    
    operation_names = []
    for path, methods in spec['paths'].items():
        for method, details in methods.items():
            if 'operationId' in details:
                operation_names.append(details['operationId'])
            else:
                # Generate operation name if operationId is missing
                name = f"{method.upper()}_{path.replace('/', '_')}"
                operation_names.append(name)
    
    return operation_names

# Function to split camelCase names into words
def split_camel_case(name):
    """Convert camelCase to individual words (e.g., 'createOrder' -> ['create', 'order'])."""
    words = []
    current_word = ''
    for char in name:
        if char.isupper() and current_word:
            words.append(current_word.lower())
            current_word = char
        else:
            current_word += char
    if current_word:
        words.append(current_word.lower())
    return words

# Step 3: Convert operation names to GloVe embeddings
def get_glove_embeddings(operation_names, model_path='glove.6B.50d.txt'):
    """Convert operation names to embeddings using a pre-trained GloVe model."""
    # Load pre-trained GloVe model (download from https://nlp.stanford.edu/projects/glove/)
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False, no_header=True)

    embeddings = []
    for name in operation_names:
        # Split camelCase names into individual words
        words = split_camel_case(name)  # e.g., 'createOrder' -> ['create', 'order']
        vectors = [word_vectors[word] for word in words if word in word_vectors]
        if vectors:
            # Average the vectors for multi-word names
            avg_vector = np.mean(vectors, axis=0)
        else:
            # Use a zero vector if no words are found
            avg_vector = np.zeros(50)  # 50 dimensions for glove.6B.50d
        embeddings.append(avg_vector)
    
    return np.array(embeddings)


def compute_dunn_index(embeddings, labels):
    """Compute the Dunn Index to evaluate clustering quality."""
    
    unique_clusters = np.unique(labels)
    
    if len(unique_clusters) < 2:
        return 0  # Dunn Index is undefined for a single cluster

    # Compute intra-cluster distances (maximum pairwise distance within a cluster)
    intra_distances = []
    for cluster in unique_clusters:
        points = embeddings[labels == cluster]
        if len(points) > 1:
            intra_distances.append(np.max(pairwise_distances(points)))
    
    max_intra_distance = max(intra_distances) if intra_distances else 0
    
    # Compute inter-cluster distances (minimum pairwise distance between clusters)
    inter_distances = []
    for i, cluster1 in enumerate(unique_clusters):
        for cluster2 in unique_clusters[i+1:]:
            points1 = embeddings[labels == cluster1]
            points2 = embeddings[labels == cluster2]
            inter_distances.append(np.min(pairwise_distances(points1, points2)))
    
    min_inter_distance = min(inter_distances) if inter_distances else 0

    return min_inter_distance / max_intra_distance if max_intra_distance != 0 else 0

# Step 4: Cluster embeddings using Affinity Propagation with grid search
def cluster_operations(embeddings):
    """Cluster embeddings and optimize parameters using grid search."""
    # Define parameter grid
    param_grid = {
        'damping': [0.5, 0.7, 0.9],
        'preference': [None, -50, -100]
    }
    
    best_score = -1
    best_labels = None
    best_params = None
    best_dbi, best_chi, best_dunn = None, None, None
    
    # Grid search over parameters
    for params in ParameterGrid(param_grid):
        af = AffinityPropagation(
            damping=params['damping'],
            preference=params['preference'],
            random_state=42
        )
        labels = af.fit_predict(embeddings)
        
        # Evaluate with silhouette score if more than one cluster
        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            dbi = davies_bouldin_score(embeddings, labels)  # Coupling
            chi = calinski_harabasz_score(embeddings, labels)  # Cohesion
            dunn = compute_dunn_index(embeddings, labels)  # Modularity Approximation
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = params
                best_dbi = dbi
                best_chi = chi
                best_dunn = dunn
    
    print(f"Best parameters: {best_params}, Silhouette Score: {best_score}")

    print(f"Silhouette Score: {best_score} (Overall clustering quality)")
    print(f"Service Coupling (Davies-Bouldin): {best_dbi} (Lower is better)")
    print(f"Service Cohesion (Calinski-Harabasz): {best_chi} (Higher is better)")
    print(f"Approximate Modularity (Dunn Index): {best_dunn} (Higher is better)")
    return best_labels

# Step 5: Group operations into microservices
def generate_microservices(operation_names, labels):
    """Group operation names by cluster labels."""
    microservices = {}
    for idx, label in enumerate(labels):
        if label not in microservices:
            microservices[label] = []
        microservices[label].append(operation_names[idx])
    
    return microservices

# Main function to execute the decomposition
def decompose_monolith_to_microservices(openapi_file, glove_model_path='glove.6B.50d.txt'):
    """Decompose a monolithic application into microservices."""
    # Extract operation names
    operation_names = load_operation_names(openapi_file)
    print("Extracted operation names:", operation_names)
    
    # Convert to embeddings
    embeddings = get_glove_embeddings(operation_names, glove_model_path)
    
    # Cluster the embeddings
    labels = cluster_operations(embeddings)
    
    # Generate microservices
    microservices = generate_microservices(operation_names, labels)
    
    # Print results
    print("\nIdentified Microservices:")
    for cluster_id, operations in microservices.items():
        print(f"Microservice {cluster_id}: {operations}")
    
    return microservices

# Example usage
if __name__ == "__main__":
    # Replace with your OpenAPI file path and GloVe model path
    openapi_file = "ecommerce.yaml"
    glove_model = "D:\\Gayatri\\BITS WILP\\Dissertation\\OpenAPI\\glove.6B\\glove.6B.50d.txt"  # Download from https://nlp.stanford.edu/projects/glove/
    
    microservices = decompose_monolith_to_microservices(openapi_file, glove_model)