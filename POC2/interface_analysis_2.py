import yaml
import numpy as np
from gensim.models import FastText
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

# Step 1 & 2: Load OpenAPI spec and extract operation names
def load_operation_names(file_path):
    """Load OpenAPI YAML file and extract operation names."""
    with open("ecommerce.yaml", "r", encoding="utf-8") as file:
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

# Step 3: Convert operation names to fastText embeddings
def get_fasttext_embeddings(operation_names, model_path='cc.en.300.bin'):
    """Convert operation names to embeddings using a pre-trained fastText model."""
    # Load pre-trained fastText model (download from https://fasttext.cc/)
    model = FastText.load_fasttext_format(model_path)
    
    # Get embeddings for each operation name
    embeddings = []
    for name in operation_names:
        embedding = model.wv[name]  # fastText handles full string with subword info
        embeddings.append(embedding)
    
    return np.array(embeddings)

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
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = params
    
    print(f"Best parameters: {best_params}, Silhouette Score: {best_score}")
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
def decompose_monolith_to_microservices(openapi_file, fasttext_model_path):
    """Decompose a monolithic application into microservices."""
    # Extract operation names
    operation_names = load_operation_names(openapi_file)
    print("Extracted operation names:", operation_names)
    
    # Convert to embeddings
    embeddings = get_fasttext_embeddings(operation_names, fasttext_model_path)
    
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
    # Replace with your OpenAPI file path and fastText model path
    openapi_file = "ecommerce.yaml"
    fasttext_model = "D:\\Gayatri\\BITS WILP\\Dissertation\\OpenAPI\\cc.en.300.bin\\cc.en.300.bin"  # Download from https://fasttext.cc/
    
    microservices = decompose_monolith_to_microservices(openapi_file, fasttext_model)