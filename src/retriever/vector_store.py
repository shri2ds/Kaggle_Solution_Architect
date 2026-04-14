import chromadb
from sentence_transformers import SentenceTransformer

class VectorIndex:
    def __init__(self, db_path="./data/chroma_db"):
        #   Initialize the Embedding Model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        #   Setup Persistent Storage
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="kaggle_knowledge")

    def add_notebook(self, notebook_name, processed_cells):
        '''
        Expects processed_cells as a list of dicts: [{'text': '...', 'metadata': {'type': 'code', ...}}, ...]
        '''

        texts = [cell['text'] for cell in processed_cells]

        #   Extract metadata and add the notebook name to it
        metadatas = []
        for cell in processed_cells:
            meta = cell['metadata']
            meta['notebook_source'] = notebook_name
            metadatas.append(meta)

        #   Generate IDs and Embeddings
        ids = [f"{notebook_name}_{i}" for i in range(len(texts))]
        embeddings = self.model.encode(texts).tolist()

        #   Add to ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ Indexed {len(texts)} cells with metadata from {notebook_name}")

    def query(self, text, cell_type=None, n_results=3):
        query_embedding = self.model.encode([text]).tolist()

        #   Adding a metadata filter
        where_filter = {"type": cell_type} if cell_type else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter  # ChromaDB filters the results before returning
        )
        return results

