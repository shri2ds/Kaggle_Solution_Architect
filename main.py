import os
from src.processor.notebook_parser import NotebookProcessor
from src.processor.chunker import SemanticChunker
from src.retriever.vector_store import VectorIndex
from src.generator.architect import KaggleArchitect


#   Parse
# processor = NotebookProcessor("notebooks/lmsys-kerasnlp-starter.ipynb")
# processed_cells = processor.parse()

#   Chunk
# chunker = SemanticChunker()
# chunks = chunker.create_chunks(raw_text)

#   Index
index = VectorIndex()
notebooks_dir = "notebooks/"
# index.add_notebook("lmsys_starter", processed_cells)

#   Looping through each notebook
for filename in os.listdir(notebooks_dir):
    if filename.endswith(".ipynb"):
        print(f"📂 Processing {filename}...")
        processor = NotebookProcessor(os.path.join(notebooks_dir, filename))
        cells = processor.parse()
        index.add_notebook(filename, cells)

#   Retrieve the top 3 snippets
query = "What models are being used here?"
results = index.query(query, cell_type="code", n_results=10)
docs = results['documents'][0]

#   DEBUG
print("--- DEBUG: WHAT THE ARCHITECT SEES ---")
for doc in docs:
    print(f">> {doc[:200]}...")

#   Initialize the Architect
architect = KaggleArchitect()

#   Generate the Final Answer
print("\n--- Kaggle Grandmaster Advice ---")
final_answer = architect.generate_strategy(query, docs)
print(final_answer)

#   Initial test code
# print("\n--- Top Matching Code Blocks ---")
# for i, doc in enumerate(results['documents'][0]):
#     print(f"\n[Match {i+1}]")
#     print("-" * 30)
#     print(doc)