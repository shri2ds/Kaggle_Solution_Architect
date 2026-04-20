# Kaggle Solution Architect 🏆 
An attempt to build specialized tools to solve high-stakes competitive problems.

**An Autonomous RAG Engine for Winning Signal Extraction**

## 🎯 Project Vision
- The Kaggle Solution Architect is a modular Retrieval-Augmented Generation (RAG) platform that automates the "Research" phase of machine learning competitions. 
- It scrapes high-performing notebooks, semantically indexes code/markdown cells, and synthesizes winning strategies using grounded LLM reasoning.

---

## 🏗️ System Architecture: The "Flywheel" Design 
The system is built on a Researcher-to-Reasoning pipeline that ensures the knowledge base stays fresh without manual intervention.

1. **The Scout (Scraper Agent):** An autonomous agent that utilizes the Kaggle CLI to monitor competition leaderboards, filtering and downloading notebooks based on voteCount and metadata.
2. **The Parser (Notebook Processor):** A specialized JSON-to-Text engine that handles the messy structure of .ipynb files, ensuring code and comments are extracted without metadata noise.
3. **The Librarian (Vector Store):** Uses Sentence-Transformers to map code logic into a vector space, stored in a persistent ChromaDB instance for sub-second similarity search.
4. **The Architect (Grounded Generator):** A strictly steered LLM (Llama-3.1) that uses Context Steering to reason over "Winning Slices," ensuring technical accuracy (e.g., distinguishing between base backbones and PEFT wrappers).

---

## 🚀 Technical Stack
- **Orchestration:** Python Subprocess (Agentic CLI Loops)
- **Vector Store:** ChromaDB (Persistent Client)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Inference Engine:** Hugging Face Inference API (Llama-3.1-8B-Instruct)
- **Data Extraction:** Kaggle API CLI / JSON Processor

---

## 📂 Repository Structure
```text
Kaggle_Solution_Architect/
├── data/                  # Persistent ChromaDB storage (GitIgnored)
├── notebooks/             # Auto-scraped .ipynb files (The Knowledge Base)
├── src/
│   ├── generator/
│   │   └── architect.py    # LLM Prompting & Grounded Logic
│   ├── processor/
│   │   ├── notebook_parser.py  # JSON notebook cell extraction
│   │   └── scraper_agent.py    # Autonomous Kaggle CLI Scout
│   └── retriever/
│       └── vector_store.py     # ChromaDB indexing & query logic
├── main.py             # Entry point for the RAG Flywheel
├── .env                # HF_TOKEN & Environment configs
├── requirements.txt    # Project dependencies
```

---

## 🧠 Key Improvements from V1
1. **Context Steering:** Implemented strict system prompts to prevent "Semantic Drift" and LLM hallucinations.
2. **Security First:** Moved from hardcoded keys to ~/.kaggle/kaggle.json and .env integration.
3. **Agentic Persistence:** The system now "closes the loop" by downloading files directly into the indexing path.
4. **Truthfulness:** Added temperature=0.0 and groundedness checks to ensure the Architect only speaks based on the data it sees.


