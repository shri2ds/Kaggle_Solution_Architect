from src.processor.notebook_parser import NotebookProcessor
from src.processor.chunker import SemanticChunker
from src.retriever.vector_store import VectorIndex

#   Parse
processor = NotebookProcessor("notebooks/lmsys-kerasnlp-starter.ipynb")
processed_cells = processor.parse()

# #   Chunk
# chunker = SemanticChunker()
# chunks = chunker.create_chunks(raw_text)

#   Index
index = VectorIndex()
index.add_notebook("lmsys_starter", processed_cells)

#   Test Query
print("\n--- Testing Semantic Search ---")
query = "How is the cross-validation handled in this notebook?"
results = index.query(query, cell_type="code")

for i, res in enumerate(results):
    print(f"\nResult {i+1}: {res[:200]}...")