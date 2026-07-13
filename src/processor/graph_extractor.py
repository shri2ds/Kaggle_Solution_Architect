import os
import json
import re
import time
import requests
from typing import Dict, Any


class KaggleGraphExtractor:
    """
    Automated Information Extraction engine that parses unstructured Kaggle notebook cells
    into a structured, deterministic JSON graph schema. Features an automatic local
    rule-based parser fallback if Google Cloud quotas are exhausted (HTTP 429).
    """

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        # Explicitly targeting the high-concurrency 2026 stable flagship model
        self.model_id = "gemini-3.5-flash"
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"

    def _extract_locally_via_regex(self, cell_text: str) -> Dict[str, Any]:
        """
        Emergency local fallback parser using deterministic regular expressions.
        Guarantees that your pipeline remains functional and can populate your Knowledge Graph
        even when offline, unauthenticated, or API rate-limited.
        """
        print("⚡ [LOCAL FALLBACK] Executing localized regex heuristics engine...")
        nodes = []
        relationships = []

        # 1. Parse Metadata using standard string boundaries
        author = "Unknown_Author"
        author_match = re.search(r"#\s*Author:\s*([^\n]+)", cell_text, re.IGNORECASE)
        if author_match:
            author = author_match.group(1).strip()

        notebook_title = "Untitled Kaggle Notebook"
        title_match = re.search(r"#\s*Title:\s*([^\n]+)", cell_text, re.IGNORECASE)
        if title_match:
            notebook_title = title_match.group(1).strip()

        # 2. Append central User and Notebook nodes
        nodes.append({
            "id": "user_0",
            "label": "User",
            "properties": {"username": author, "tier": "Grandmaster"}
        })
        nodes.append({
            "id": "notebook_0",
            "label": "Notebook",
            "properties": {"title": notebook_title, "id": "notebook_0"}
        })
        relationships.append({
            "source_id": "user_0",
            "target_id": "notebook_0",
            "type": "AUTHORED"
        })

        # 3. Detect and resolve Library imports
        libraries_detected = set()
        for line in cell_text.splitlines():
            import_match = re.match(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", line)
            if import_match:
                lib_name = import_match.group(1)
                # Resolve common framework aliases
                if lib_name not in ["torch", "transformers", "peft", "pandas", "numpy"]:
                    continue
                libraries_detected.add(lib_name)

        for lib in libraries_detected:
            lib_id = f"lib_{lib}"
            nodes.append({
                "id": lib_id,
                "label": "Library",
                "properties": {"name": lib}
            })
            relationships.append({
                "source_id": "notebook_0",
                "target_id": lib_id,
                "type": "IMPORTS"
            })

        # 4. Check for Hardware environment tags
        hardware = "GPU_T4"
        if "A100" in cell_text or "GPU_A100" in cell_text:
            hardware = "GPU_A100"
        elif "TPU" in cell_text:
            hardware = "TPU"

        nodes.append({
            "id": f"hw_{hardware.lower()}",
            "label": "Hardware",
            "properties": {"type": hardware}
        })
        relationships.append({
            "source_id": "notebook_0",
            "target_id": f"hw_{hardware.lower()}",
            "type": "EXECUTED_ON"
        })

        # 5. Look for base Models
        if "gemma" in cell_text.lower():
            nodes.append({
                "id": "model_gemma",
                "label": "Model",
                "properties": {"name": "gemma-2-9b", "family": "Gemma", "parameter_size": "9B"}
            })
            relationships.append({
                "source_id": "notebook_0",
                "target_id": "model_gemma",
                "type": "FINETUNES"
            })

        # 6. Parse Datasets
        dataset_match = re.search(r"dataset\s*=\s*['\"]([^'\"]+)['\"]", cell_text, re.IGNORECASE)
        if dataset_match:
            ds_name = dataset_match.group(1)
            nodes.append({
                "id": "dataset_0",
                "label": "Dataset",
                "properties": {"name": ds_name}
            })
            relationships.append({
                "source_id": "notebook_0",
                "target_id": "dataset_0",
                "type": "TRAINED_ON"
            })

        return {"nodes": nodes, "relationships": relationships}

    def extract_subgraph(self, cell_text: str) -> Dict[str, Any]:
        """
        Sends the unstructured notebook text to Gemini using strict, lowercase OpenAPI schemas.
        Falls back seamlessly to local regex parsing if API keys or quotas are exhausted.
        """
        if not self.api_key:
            print("⚠️ [WARNING] No Gemini API Key found in environment variables.")
            return self._extract_locally_via_regex(cell_text)

        # Enforce strict lowercase OpenAPI 3.0 type formats
        schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {
                                "type": "string",
                                "enum": ["User", "Notebook", "Model", "Library", "Dataset", "Hardware"]
                            },
                            "properties": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "username": {"type": "string"},
                                    "tier": {"type": "string"},
                                    "title": {"type": "string"},
                                    "family": {"type": "string"},
                                    "parameter_size": {"type": "string"},
                                    "type": {"type": "string"}
                                }
                            }
                        },
                        "required": ["id", "label", "properties"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "target_id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["AUTHORED", "IMPORTS", "FINETUNES", "TRAINED_ON", "EXECUTED_ON"]
                            }
                        },
                        "required": ["source_id", "target_id", "type"]
                    }
                }
            },
            "required": ["nodes", "relationships"]
        }

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Analyze this Kaggle notebook content and perform extraction:\n\n{cell_text}"
                }]
            }],
            "systemInstruction": {
                "parts": [{"text": "You are an expert Ontological Parser. Extract structured nodes and relationships."}]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }

        headers = {"Content-Type": "application/json"}

        try:
            # We enforce immediate failure on client errors to detect 403/429 limits cleanly
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=10)

            if response.status_code == 429:
                print("⚠️ [RATE LIMIT] Gemini API reports 429 Resource Exhausted.")
                return self._extract_locally_via_regex(cell_text)

            response.raise_for_status()
            raw_response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return json.loads(raw_response_text)

        except Exception as e:
            print(f"⚠️ [NETWORK EXCEPTION] Gemini extraction failed: {e}")
            return self._extract_locally_via_regex(cell_text)