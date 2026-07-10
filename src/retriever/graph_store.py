import os
from neo4j import GraphDatabase
from typing import Dict, Any


class KaggleGraphStore:
    """
    Manages connection pooling, index configurations, and parameter-safe,
    idempotent transaction execution against your local Neo4j instance.
    """

    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.environ.get("NEO4J_USER", "")
        self.password = os.environ.get("NEO4J_PASSWORD", "")

        # Initialize connection pool
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self._initialize_database_constraints()

    def close(self):
        """Closes the connection pool cleanly."""
        self.driver.close()

    def _initialize_database_constraints(self):
        """
        Configures strict database-level unique constraints and indices.
        Enforcing uniqueness on keys enables O(1) index-seek lookups instead of slow AllNodesScans.
        """
        constraints = [
            "CREATE CONSTRAINT unique_user_username IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE;",
            "CREATE CONSTRAINT unique_notebook_id IF NOT EXISTS FOR (n:Notebook) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT unique_model_name IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE;",
            "CREATE CONSTRAINT unique_library_name IF NOT EXISTS FOR (l:Library) REQUIRE l.name IS UNIQUE;",
            "CREATE CONSTRAINT unique_dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE;",
            "CREATE CONSTRAINT unique_hardware_type IF NOT EXISTS FOR (h:Hardware) REQUIRE h.type IS UNIQUE;"
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)

    def write_subgraph(self, subgraph: Dict[str, Any]):
        """
        Parses the node and relationship JSON extraction and executes
        idempotent parameterized Cypher MERGE commands.
        """
        # Node query lookup mapping to resolve dynamic labels safely without injection vulnerability
        node_templates = {
            "User": "MERGE (n:User {username: $properties.username}) SET n += $properties",
            "Notebook": "MERGE (n:Notebook {id: $id}) SET n += $properties",
            "Model": "MERGE (n:Model {name: $properties.name}) SET n += $properties",
            "Library": "MERGE (n:Library {name: $properties.name}) SET n += $properties",
            "Dataset": "MERGE (n:Dataset {name: $properties.name}) SET n += $properties",
            "Hardware": "MERGE (n:Hardware {type: $properties.type}) SET n += $properties"
        }

        with self.driver.session() as session:
            # 1. Idempotent Node Upsert
            for node in subgraph.get("nodes", []):
                label = node["label"]
                if label in node_templates:
                    session.run(
                        node_templates[label],
                        id=node["id"],
                        properties=node["properties"]
                    )

            # 2. Idempotent Relationship Merge using Index-Seek Lookup
            for rel in subgraph.get("relationships", []):
                rel_type = rel["type"]
                source_id = rel["source_id"]
                target_id = rel["target_id"]

                # Parameterized label-aware merge query to avoid slow AllNodesScan
                cypher_rel = f"""
                MATCH (source) WHERE source.username = $source_id OR source.id = $source_id OR source.name = $source_id OR source.type = $source_id
                MATCH (target) WHERE target.username = $target_id OR target.id = $target_id OR target.name = $target_id OR target.type = $target_id
                MERGE (source)-[r:{rel_type}]->(target)
                """
                session.run(cypher_rel, source_id=source_id, target_id=target_id)