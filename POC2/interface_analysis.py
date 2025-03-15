import yaml
import json
import requests
import spacy
import nltk
from nltk.corpus import wordnet as wn
from fuzzywuzzy import process

# Load NLP model
nlp = spacy.load("en_core_web_md")
nltk.download("wordnet")

# Fetch Schema.org vocabulary (limited to core classes)
SCHEMA_ORG_URL = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
response = requests.get(SCHEMA_ORG_URL)
schema_org_classes = []



if response.status_code == 200:
    data = response.json()
    for item in data["@graph"]:
        if item["@type"] == "rdfs:Class":
            schema_org_classes.append(item["@id"].split("/")[-1].lower())  # Extract class name

print(schema_org_classes)

# Function to find best Schema.org match
def find_best_match(term):
    term = term.lower()
    
    # 1. Exact match
    if term in schema_org_classes:
        return f"schema:{term}"
    
    # 2. Fuzzy matching
    best_match, score = process.extractOne(term, schema_org_classes)
    if score > 80:  # Set threshold for similarity
        return f"schema:{best_match}"
    
    # 3. WordNet similarity
    term_synsets = wn.synsets(term)
    if term_synsets:
        for schema_term in schema_org_classes:
            schema_synsets = wn.synsets(schema_term)
            if schema_synsets:
                similarity = term_synsets[0].wup_similarity(schema_synsets[0])
                if similarity and similarity > 0.7:
                    return f"schema:{schema_term}"
    
    return "No match"

# Load OpenAPI spec from file
def load_openapi_spec(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Extract key terms from OpenAPI spec
def extract_api_terms(openapi_spec):
    extracted_terms = {}

    # Extract paths
    for path, methods in openapi_spec.get("paths", {}).items():
        path_key = path.strip("/").split("/")[0]  # Take main entity name
        extracted_terms[path] = {"concept": find_best_match(path_key), "properties": {}}
        
        # Extract parameters
        for method, details in methods.items():
            if "parameters" in details:
                for param in details["parameters"]:
                    if "name" in param:
                        param_name = param["name"]
                        extracted_terms[path]["properties"][param_name] = find_best_match(param_name)
    
    # # Extract schemas
    # for schema_name, schema_details in openapi_spec.get("components", {}).get("schemas", {}).items():
    #     extracted_terms[schema_name] = {"concept": find_best_match(schema_name), "properties": {}}
    #     if "properties" in schema_details:
    #         for prop_name in schema_details["properties"]:
    #             extracted_terms[schema_name]["properties"][prop_name] = find_best_match(prop_name)

    return extracted_terms

# Main execution

import requests
import yaml

# URL of the raw OpenAPI spec file
url = "https://raw.githubusercontent.com/swagger-api/swagger-petstore/refs/heads/master/src/main/resources/openapi.yaml"

# Fetch and save the file locally
# response = requests.get(url)
# if response.status_code == 200:
#     with open("petstore_openapi.yaml", "w", encoding="utf-8") as f:
#         f.write(response.text)
# else:
#     print(f"Error: Unable to download file, status code {response.status_code}")



# Compute similarity scores
def compute_similarity(api1, api2):
    concept1, concept2 = mappings[api1]["concept"], mappings[api2]["concept"]
    props1, props2 = set(mappings[api1]["properties"].values()), set(mappings[api2]["properties"].values())

    # Concept similarity (1 if same, 0 if different)
    concept_similarity = 1 if concept1 == concept2 else 0

    # Property overlap (Jaccard similarity)
    property_similarity = len(props1 & props2) / len(props1 | props2) if props1 | props2 else 0

    # Final similarity score (weighted sum)
    return 0.7 * concept_similarity + 0.3 * property_similarity

import networkx as nx
import json
#import community  # Louvain algorithm
import numpy as np
from itertools import combinations
import community.community_louvain as community

# Load OpenAPI spec from file
with open("ecommerce.yaml", "r", encoding="utf-8") as file:
    openapi_spec = yaml.safe_load(file)
    print("YAML Loaded Successfully!")
    mappings = extract_api_terms(openapi_spec)

    # Print result
    print(json.dumps(mappings, indent=2))



# # Sample extracted API mappings from OpenAPI
# mappings = {
#     "/users/{id}": {"concept": "schema:Person", "properties": {"name": "schema:name", "email": "schema:email"}},
#     "/profiles/{id}": {"concept": "schema:Person", "properties": {"username": "schema:identifier"}},
#     "/orders/{id}": {"concept": "schema:Order", "properties": {"amount": "schema:price", "status": "schema:OrderStatus"}},
#     "/products/{id}": {"concept": "schema:Product", "properties": {"name": "schema:name", "price": "schema:price"}}
# }

# Build a similarity graph
    G = nx.Graph()

    # Add nodes (APIs)
    for api in mappings.keys():
        G.add_node(api, concept=mappings[api]["concept"])



    # Add edges based on similarity
    threshold = 0.5  # Define a threshold for similarity
    for api1, api2 in combinations(mappings.keys(), 2):
        similarity = compute_similarity(api1, api2)
        if similarity > threshold:
            G.add_edge(api1, api2, weight=similarity)

    # Apply Louvain clustering for microservice grouping
    partition = community.best_partition(G, weight="weight")

    # Organize clusters
    microservices = {}
    for api, cluster in partition.items():
        if cluster not in microservices:
            microservices[cluster] = []
        microservices[cluster].append(api)

    # Print the final microservice groupings
    print(json.dumps(microservices, indent=2))




