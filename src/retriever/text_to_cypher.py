import os
import re
import json
import requests
from typing import List, Dict, Any
from Kaggle_Solution_Architect.src.retriever.graph_store import KaggleGraphStore

class TextToCypherAgent:
    """
    Agentic retrieval component that translates natural language queries into
    executable Cypher graph queries. Enforces strict schema grounding and read-only
    sanitizations to protect against Cypher injection attacks.
    """

    # Official Ontology Schema for System Instruction
    SCHEMA_DEFINITION = """
    Node Labels and Properties:
    - User {username: STRING, tier: STRING}
    - Notebook {id: STRING, title: STRING, views: INTEGER, votes: INTEGER}
    - Model {name: STRING, family: STRING, parameter_size: STRING}
    - Library {name: STRING}
    - Dataset {name: STRING}
    - Hardware {type: STRING}

    Directed Relationships:
    - (:User)-[:AUTHORED]->(:Notebook)
    - (:Notebook)-[:IMPORTS]->(:Library)
    - (:Notebook)-[:FINETUNES]->(:Model)
    - (:Notebook)-[:TRAINED_ON]->(:Dataset)
    - (:Notebook)-[:EXECUTED_ON]->(:Hardware)
    """

    FORBIDDEN_KEYWORDS = [
        "DELETE", "DETACH", "CREATE", "MERGE", "SET",
        "REMOVE", "DROP", "CALL", "APOC", "DB"
    ]

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model_id = "gemini-3.5-flash"
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
        self.store = KaggleGraphStore()

    def _sanitize_cypher(self, raw_cypher: str) -> str:
        """
        Cleans LLM response output and enforces strict read-only constraints.
        Prevents destructive Cypher injection queries.
        """
        # Strip markdown code blocks if the model included them
        clean_code = re.sub(r"```(?:cypher)?", "", raw_cypher, flags=re.IGNORECASE).strip()
        clean_code = clean_code.replace("```", "").strip()

        # Check for non-read-only keywords
        upper_code = clean_code.upper()
        for kw in self.FORBIDDEN_KEYWORDS:
            # Match whole-word keywords to prevent false positives in property names
            if re.search(r"\b" + kw + r"\b", upper_code):
                raise ValueError(
                    f"⚠️ Security Violation: Query contained forbidden non-read-only keyword '{kw}'."
                )

        if "MATCH" not in upper_code or "RETURN" not in upper_code:
            raise ValueError("⚠️ Invalid Query: Cypher query must contain MATCH and RETURN statements.")

        return clean_code

    def _fallback_heuristic_cypher(self, natural_query: str) -> str:
        """
        Deterministic regex/keyword fallback when API limits are reached.
        """
        query_lower = natural_query.lower()

        if "library" in query_lower or "libraries" in query_lower or "import" in query_lower:
            return "MATCH (n:Notebook)-[:IMPORTS]->(l:Library) RETURN n.title AS notebook, collect(l.name) AS libraries LIMIT 10;"
        elif "model" in query_lower or "fine-tune" in query_lower or "gemma" in query_lower:
            return "MATCH (n:Notebook)-[:FINETUNES]->(m:Model) RETURN n.title AS notebook, m.name AS model LIMIT 10;"
        elif "author" in query_lower or "user" in query_lower or "grandmaster" in query_lower:
            return "MATCH (u:User)-[:AUTHORED]->(n:Notebook) RETURN u.username AS author, u.tier AS tier, n.title AS notebook LIMIT 10;"
        elif "hardware" in query_lower or "gpu" in query_lower or "a100" in query_lower:
            return "MATCH (n:Notebook)-[:EXECUTED_ON]->(h:Hardware) RETURN n.title AS notebook, h.type AS hardware LIMIT 10;"
        else:
            return "MATCH (u:User)-[:AUTHORED]->(n:Notebook)-[:FINETUNES]->(m:Model) RETURN u.username AS author, n.title AS notebook, m.name AS model LIMIT 10;"

    def generate_cypher(self, natural_query: str) -> str:
        """
        Converts natural language input to grounded Cypher statements.
        """
        if not self.api_key:
            print("⚡ [OFFLINE MODE] No API key detected. Utilizing deterministic rule engine...")
            return self._fallback_heuristic_cypher(natural_query)

        prompt = f"""
        Given the following Neo4j graph database ontology schema:
        {self.SCHEMA_DEFINITION}

        Task: Translate this natural language user query into a valid Cypher query:
        "{natural_query}"

        Rules:
        1. Return ONLY the raw Cypher statement. Do NOT include markdown code blocks, intro text, or explanations.
        2. Execute ONLY read-only queries starting with MATCH and ending with RETURN.
        3. Do NOT invent labels, relationship types, or properties not listed in the schema.
        4. Always alias returned values cleanly (e.g., RETURN u.username AS author, n.title AS notebook).
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {
                "parts": [{
                              "text": "You are a specialized Text-to-Cypher translator. Generate precise, single-statement Cypher queries."}]
            }
        }

        try:
            res = requests.post(self.endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
            if res.status_code == 429:
                print("⚠️ [RATE LIMIT] Exceeded quota. Switching to offline fallback engine...")
                return self._fallback_heuristic_cypher(natural_query)

            res.raise_for_status()
            raw_text = res.json()['candidates'][0]['content']['parts'][0]['text']
            return self._sanitize_cypher(raw_text)
        except Exception as e:
            print(f"⚠️ [EXTRACTION EXCEPTION] Text-to-Cypher generation failed: {e}")
            return self._fallback_heuristic_cypher(natural_query)

    def execute_query(self, natural_query: str) -> Dict[str, Any]:
        """
        Translates natural language to Cypher, validates security constraints,
        and executes the query against the Neo4j graph database.
        """
        cypher_query = self.generate_cypher(natural_query)
        print(f"\n🔍 [GENERATED CYPHER]: {cypher_query}")

        try:
            with self.store.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                return {
                    "natural_query": natural_query,
                    "cypher_query": cypher_query,
                    "results_count": len(records),
                    "data": records
                }
        except Exception as e:
            return {
                "natural_query": natural_query,
                "cypher_query": cypher_query,
                "error": f"Execution failed: {str(e)}"
            }

    def close(self):
        self.store.close()
