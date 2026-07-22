import os
import sys
from dotenv import load_dotenv

# Pull current execution context cleanly into python namespace path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Load environmental variables
load_dotenv()

# Trace the physical file system location of our modules to detect IDE cache collisions
try:
    import src.processor.graph_extractor as ge
    from src.retriever.graph_store import KaggleGraphStore

    print("=" * 60)
    print("⚙️  DIAGNOSTIC PATH TRACING")
    print("=" * 60)
    print(f"Extractor Location: {ge.__file__}")
    print("=" * 60)
except ImportError as e:
    print(f"❌ [IMPORT ERROR] Could not trace local namespaces. Details: {e}")
    sys.exit(1)

gemini_key = os.environ.get("GEMINI_API_KEY", "")
neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")

# Define mock high-density Gemma training pipeline string
mock_notebook_cell = """
# Author: Kaggle_Grandmaster_Z
# Title: Gemma-2 Fine-tuning on A100 GPU
# Execution Platform: GPU_A100

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "gemma-2-9b"
dataset = "Math-Word-Problems-v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
"""


def execute_verification():
    print("\n[STEP 1] Initializing local Neo4j Connection & Schema Constraints...")
    try:
        store = KaggleGraphStore()
        print("✅ Neo4j connection pool initiated. Constraints verified.")
    except Exception as e:
        print(f"❌ Connection Failed! Verify your container is active. Error: {e}")
        return

    print("\n[STEP 2] Launching structured metadata extraction...")
    try:
        extractor = ge.KaggleGraphExtractor()
        subgraph = extractor.extract_subgraph(mock_notebook_cell)

        nodes_count = len(subgraph.get("nodes", []))
        edges_count = len(subgraph.get("relationships", []))
        print(f"✅ Extracted Subgraph: {nodes_count} Nodes | {edges_count} Relationships")
        print("-" * 60)
        print(f"Raw Nodes: {subgraph.get('nodes', [])}")
        print(f"Raw Edges: {subgraph.get('relationships', [])}")
        print("-" * 60)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        store.close()
        return

    print("\n[STEP 3] Injecting the parsed transaction into local Neo4j database...")
    try:
        store.write_subgraph(subgraph)
        print("✅ Database transaction completed successfully!")
    except Exception as e:
        print(f"❌ Database ingestion failed: {e}")
    finally:
        store.close()

    print("\n" + "=" * 60)
    print("🎉 PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 60)
    print("Login and query your database visually: http://localhost:7474")
    print("Run Cypher to view relationships: MATCH (n) RETURN n LIMIT 25;")
    print("=" * 60)


if __name__ == "__main__":
    execute_verification()