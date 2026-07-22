from typing import List, Dict, Any
import os
from Kaggle_Solution_Architect.src.retriever.graph_store import KaggleGraphStore

class KaggleHybridRetriever:
    """
    Orchestrates the hybrid retrieval loop:
    1. Performs semantic vector queries against ChromaDB to find the most relevant code chunks.
    2. Uses metadata keys to jump directly into Neo4j.
    3. Traverses the graph to extract structured context (Author, Libraries, Model, Dataset, Hardware).
    4. Merges both contexts into a unified, high-density context block.
    """

    def __init__(self, vector_store=None):
        # We accept an instantiated vector store to maintain separation of concerns
        self.vector_store = vector_store
        self.graph_store = KaggleGraphStore()

    def _retrieve_graph_context(self, notebook_id: str) -> Dict[str, Any]:
        """
        Runs an optimized, index-seek Cypher query to retrieve the complete
        ontological neighborhood of a target Notebook.
        """
        query = """
        MATCH (n:Notebook {id: $notebook_id})
        OPTIONAL MATCH (u:User)-[:AUTHORED]->(n)
        OPTIONAL MATCH (n)-[:IMPORTS]->(l:Library)
        OPTIONAL MATCH (n)-[:FINETUNES]->(m:Model)
        OPTIONAL MATCH (n)-[:TRAINED_ON]->(d:Dataset)
        OPTIONAL MATCH (n)-[:EXECUTED_ON]->(h:Hardware)
        RETURN 
            n.title AS title,
            u.username AS author,
            u.tier AS author_tier,
            collect(DISTINCT l.name) AS libraries,
            collect(DISTINCT m.name) AS models,
            collect(DISTINCT d.name) AS datasets,
            h.type AS hardware
        """

        with self.graph_store.driver.session() as session:
            result = session.run(query, notebook_id=notebook_id)
            record = result.single()
            if record:
                return {
                    "title": record["title"],
                    "author": record["author"],
                    "author_tier": record["author_tier"],
                    "libraries": record["libraries"],
                    "models": record["models"],
                    "datasets": record["datasets"],
                    "hardware": record["hardware"]
                }
            return {}

    def retrieve_hybrid_context(self, query_text: str, n_results: int = 1) -> str:
        """
        Executes the hybrid vector + graph context extraction loop.
        """
        # Step 1: Query ChromaDB for semantic search matches
        if not self.vector_store:
            raise RuntimeError("ChromaDB vector store is not initialized on this hybrid retriever.")

        vector_results = self.vector_store.query(query_text, n_results=n_results)

        # Ensure we have valid matches
        if not vector_results or 'documents' not in vector_results or not vector_results['documents'][0]:
            return "No matching records found in local index."

        # Extract the highest scoring code chunk and its notebook metadata ID
        semantic_chunk = vector_results['documents'][0][0]
        metadata = vector_results['metadatas'][0][0] if 'metadatas' in vector_results else {}
        notebook_id = metadata.get("notebook_id", "")

        # Step 2: Jump directly to Neo4j if metadata points to a valid notebook
        graph_context_str = ""
        if notebook_id:
            graph_data = self._retrieve_graph_context(notebook_id)
            if graph_data:
                graph_context_str = (
                    f"--- RELATION METADATA BACKGROUND ---\n"
                    f"Notebook: '{graph_data.get('title')}' (ID: {notebook_id})\n"
                    f"Author: {graph_data.get('author')} (Tier: {graph_data.get('author_tier')})\n"
                    f"Imported Libraries: {', '.join(graph_data.get('libraries', []))}\n"
                    f"Fine-Tuned Models: {', '.join(graph_data.get('models', []))}\n"
                    f"Datasets Used: {', '.join(graph_data.get('datasets', []))}\n"
                    f"Hardware Environment: {graph_data.get('hardware')}\n"
                    f"------------------------------------\n"
                )

        # Step 3: Compile the final rich context block
        rich_context = (
            f"{graph_context_str}"
            f"--- CODE BLOCK CONTENT ---\n"
            f"{semantic_chunk}\n"
            f"--------------------------"
        )
        return rich_context

    def close(self):
        """Cleans up the database pool connections."""
        self.graph_store.close()
