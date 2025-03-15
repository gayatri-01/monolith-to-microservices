import os
import re
import networkx as nx
import matplotlib.pyplot as plt

# Path to the monolithic Java project
PROJECT_PATH = "../Monolith/src/main/java/com/monolith/poc/"

# Regular expressions to detect dependencies
IMPORT_REGEX = re.compile(r"import\s+com\.monolith\.poc\.(model|repository|service|controller)\.(\w+);")
CLASS_REGEX = re.compile(r"public\s+(class|interface)\s+(\w+)")


def extract_dependencies():
    # Create a directed graph
    dependency_graph = nx.DiGraph()
    """Extract dependencies from Java files and build a dependency graph."""
    for root, _, files in os.walk(PROJECT_PATH):
        for file in files:
            if (file.endswith("Application.java")):
                continue
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                print(f"Processing file {file_path}")
                # Extract class name
                class_match = CLASS_REGEX.search(content)
                if class_match:
                    class_name = class_match.group(2)
                    print(f"Adding node {class_name}")
                    dependency_graph.add_node(class_name)

                    # Extract dependencies
                    imports = IMPORT_REGEX.findall(content)
                    print(f"Found dependencies: {imports}")
                    for package, dependency in imports:
                        print(f"Adding edge {class_name}-{dependency}")
                        dependency_graph.add_edge(class_name, dependency)
    
    # Visualize Dependency Graph
    plt.figure(figsize=(10, 7))
    nx.draw(dependency_graph, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.title("Monolith Dependency Graph")
    plt.show()

    return dependency_graph

                
extract_dependencies()


