from fastapi import FastAPI
import os
import re
import networkx as nx
from fastapi.middleware.cors import CORSMiddleware

PROJECT_PATH = "../Monolith/src/main/java/com/monolith/poc/"
IMPORT_REGEX = re.compile(r"import\s+com\.monolith\.poc\.(model|repository|service|controller)\.(\w+);")
CLASS_REGEX = re.compile(r"public\s+(class|interface)\s+(\w+)")

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_dependencies():
    dependency_graph = nx.DiGraph()
    nodes = []
    edges = []

    for root, _, files in os.walk(PROJECT_PATH):
        for file in files:
            if file.endswith("Application.java"):
                continue
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                class_match = CLASS_REGEX.search(content)
                if class_match:
                    class_name = class_match.group(2)
                    dependency_graph.add_node(class_name)
                    nodes.append({"id": class_name, "label": class_name, "color": "#70a1ff"})

                    imports = IMPORT_REGEX.findall(content)
                    for package, dependency in imports:
                        dependency_graph.add_edge(class_name, dependency)
                        edges.append({"source": class_name, "target": dependency})

    return {"nodes": nodes, "edges": edges}

@app.get("/graph-data")
def get_graph_data():
    return extract_dependencies()
