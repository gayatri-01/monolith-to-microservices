import csv
import numpy as np
import pandas as pd

#Both of these CSV work well - Ollama & Louvain are used

# CSV_FILE = "D:\Gayatri\BITS WILP\Dissertation\POC2\code_vectors.csv"
CSV_FILE = "D:\\Gayatri\\BITS WILP\\Dissertation\\POC2\\code_vectors_blog.csv"

def load_code_vectors():
    """
    Reads function vectors from a CSV file and returns function names and vectors.
    """
    function_names = []
    code_vectors = []

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            function_name = row[0]  # First column is function name
            vector = np.array(row[1:], dtype=np.float32)  # Convert remaining columns to float
            function_names.append(function_name)
            code_vectors.append(vector)

    return function_names, np.array(code_vectors)  # Return function names and vectors

# Example usage
function_names, code_vectors = load_code_vectors()

from sklearn.preprocessing import normalize
code_vectors = normalize(code_vectors)


print(f"Loaded {len(function_names)} function vectors.")
# print("Example function:", function_names[0])
# print("Example vector:", code_vectors[0])


df = pd.DataFrame(code_vectors, index=function_names)
print(df.head())  # Check if all functions are present
print("Total Functions in CSV:", df.shape[0])


from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
#import community as community_louvain  # pip install python-louvain
import community.community_louvain as community


# Compute cosine similarity between all function vectors
#similarity_matrix = cosine_similarity(df.values)

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

similarity_matrix = cosine_similarity(code_vectors)
sim_df = pd.DataFrame(similarity_matrix, index=function_names, columns=function_names)

print("\nFunction Similarity Matrix:")
print(sim_df)

import community as community_louvain  

# Create graph
G = nx.Graph()
for i, func in enumerate(function_names):
    for j, func2 in enumerate(function_names):
        if i != j:
            if similarity_matrix[i][j] > 0.2: 
                G.add_edge(function_names[i], function_names[j], weight=similarity_matrix[i][j])

print("Functions in Graph:", set(G.nodes))


# Apply Louvain Clustering with resolution tuning
# partition = community_louvain.best_partition(G, resolution=0.8) # Higher = fewer clusters
partition = community.best_partition(G, resolution=0.8)

# Convert to microservices
microservices = {}
for func, cluster_id in partition.items():
    if cluster_id not in microservices:
        microservices[cluster_id] = []
    microservices[cluster_id].append(func)


###### Metrics #############

from sklearn.metrics import silhouette_score
import numpy as np

# Extract function embeddings from the similarity matrix
emb_array = np.array([similarity_matrix[i] for i, func in enumerate(function_names) if func in G.nodes])

# Create label mapping for partitioned clusters
labels = np.array([partition[func] for func in function_names if func in partition])

# Compute Silhouette Score
silhouette = silhouette_score(emb_array, labels)
print(f"Silhouette Score: {silhouette:.4f}")

import networkx as nx
import community

# Extract communities from partition
communities = [set(funcs) for cluster_id, funcs in microservices.items()]

# Compute Modularity Score
modularity_score = nx.algorithms.community.quality.modularity(G, communities)
print(f"Modularity Score: {modularity_score:.4f}")


# Compute Service Coupling (Inter-service calls / Total calls)
inter_service_calls = 0
total_calls = 0

for func in G.nodes:
    for neighbor in G.neighbors(func):
        total_calls += 1  # Count every function call (edge)
        if partition[func] != partition[neighbor]:  # If functions are in different services
            inter_service_calls += 1  

# Normalize coupling score
service_coupling = inter_service_calls / total_calls if total_calls > 0 else 0
print(f"Service Coupling Score: {service_coupling:.4f}")

# Compute Service Cohesion (Intra-service calls / Total calls in the service)
cohesion_scores = []
for cluster_id, funcs in microservices.items():
    intra_service_calls = 0
    service_total_calls = 0

    for func in funcs:
        for neighbor in G.neighbors(func):
            if neighbor in funcs:
                intra_service_calls += 1  # Count internal function calls
            service_total_calls += 1  # Count all calls involving this service

    cohesion_score = intra_service_calls / service_total_calls if service_total_calls > 0 else 0
    cohesion_scores.append(cohesion_score)

# Take the average cohesion across all microservices
service_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0
print(f"Service Cohesion Score: {service_cohesion:.4f}")



############################

# Display clusters
for cluster_id, functions in microservices.items():
    print(f"Louvain-Microservice {cluster_id}: {functions}")



import openai  # Or use any LLM API
import re
import spacy
from collections import Counter

# Load spaCy for NLP preprocessing
nlp = spacy.load("en_core_web_sm")

# Example microservice clusters
# microservices = {
#     0: ['getSiteName', 'setSiteName', 'getPageSize', 'setPageSize', 'getSiteSlogan', 'setSiteSlogan', 'getApplicationEnv', 'configure', 'addCorsMappings'],
#     1: ['findOrCreateByName', 'getTag', 'getAllTags', 'put', 'get', 'deleteTag', 'getTagNames'],
#     2: ['getSuperUser', 'signin', 'createUser', 'currentUser', 'preHandle', 'postHandle', 'viewObjectAddingInterceptor', 'parseTagNames', 'authenticate', 'createSpringUser', 'addInterceptors', 'changePassword', 'createAuthority', 'loadUserByUsername', 'rememberMeServices', 'passwordEncoder'],
#     3: ['getPost', 'getPublishedPostByPermalink', 'createPost', 'updatePost', 'deletePost', 'getArchivePosts', 'getPostTags', 'extractPostMeta', 'getAllPublishedPostsByPage', 'createAboutPage', 'findPostsByTag', 'countPostsByTags', 'incrementViews'],
#     4: ['initialize', 'registerJadeViewHelpers']
# }

# Function to generate a meaningful microservice name using OpenAI (or any LLM API)

from transformers import pipeline
# 


# from transformers import pipeline

# # Load a lightweight local LLM (or use OpenAI API if needed)
# chatbot = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# def generate_service_name(functions):
#     """
#     Generates a concise and meaningful service name based on function names.
#     """
#     prompt = (
#         "You are an expert in software architecture. Given the following function names, "
#         "suggest a meaningful microservice name:\n\n"
#         f"Functions: {', '.join(functions)}\n\n"
#         "The name should be concise, descriptive, and follow naming conventions like 'UserService', "
#         "'OrderProcessingService', etc.\n\n"
#         "Output only the name with no explanation."
#     )

#     # Generate response
#     response = chatbot(prompt, max_new_tokens=10, truncation=True)
#     return response

# Automatically name microservices using LLM
# named_microservices = {
#     generate_microservice_name(functions): functions
#     for cluster_id, functions in microservices.items()
# }

# # Print the automatically generated service names
# for service_name, functions in named_microservices.items():
#     print(f"{service_name}: {functions}")





import openai

#openai.api_key = "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"  # Replace with actual API key

from transformers import pipeline

# Load an open-source LLM (Mistral-7B is good for reasoning)
#chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", device_map="auto",api_key="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")
import ollama

def generate_service_name(functions):
    prompt = (
        "You are an expert in software architecture. Given the following function names, "
        "suggest a meaningful microservice name:\n\n"
        f"Functions: {', '.join(functions)}\n\n"
        "The name should be concise, descriptive, and follow naming conventions like 'UserService', "
        "'OrderProcessingService', etc.\n\n"
        "Output only the name with no explanation. IMPORTANT - ONLY ONE WORD RESPONSE!!!!"
    )

    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()





# for ms, functions in microservices.items():
#     print(f"{ms}: {generate_service_name(functions)}")


# KMeans
# from sklearn.cluster import KMeans

# NUM_CLUSTERS = 4  # Adjust as needed
# # kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
# kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
# cluster_labels = kmeans.fit_predict(df.values)

# microservices = {}
# for func, cluster_id in zip(function_names, cluster_labels):
#     if cluster_id not in microservices:
#         microservices[cluster_id] = []
#     microservices[cluster_id].append(func)

# for cluster_id, functions in microservices.items():
#     print(f"KMeans-Microservice {cluster_id}: {functions}")

# from sklearn.cluster import DBSCAN
# Apply DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
# labels = dbscan.fit_predict(code_vectors)

# dbscan = DBSCAN(eps=0.6, min_samples=1, metric='cosine')  # Lower eps, min_samples
# labels = dbscan.fit_predict(code_vectors)

# # Group Functions into Microservices
# microservices = {}
# for i, label in enumerate(labels):
#     if label not in microservices:
#         microservices[label] = []
#     microservices[label].append(function_names[i])

# # Display Clusters
# for cluster_id, functions in microservices.items():
#     print(f"DBSCAN-Microservice {cluster_id}: {functions}")