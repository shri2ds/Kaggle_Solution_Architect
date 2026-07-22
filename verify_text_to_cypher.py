import os
import sys
from dotenv import load_dotenv

# Ensure current execution context is in python namespace
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

from src.retriever.text_to_cypher import TextToCypherAgent


def run_text_to_cypher_verification():
    print("=" * 65)
    print("🕸️ DAY 27: AGENTIC TEXT-TO-CYPHER PIPELINE VERIFICATION")
    print("=" * 65)

    agent = TextToCypherAgent()

    test_queries = [
        "Which authors have fine-tuned Gemma models?",
        "List all libraries imported by Kaggle notebooks.",
        "What hardware was used for running fine-tuning experiments?"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n--- [TEST {idx}] Natural Query: '{query}' ---")
        output = agent.execute_query(query)

        if "error" in output:
            print(f"❌ Error: {output['error']}")
        else:
            print(f"✅ Executed Successfully! Matches Found: {output['results_count']}")
            print("Data Payload:")
            for item in output['data']:
                print(f"  └─ {item}")

    agent.close()
    print("\n" + "=" * 65)
    print("🎉 TEXT-TO-CYPHER VERIFICATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_text_to_cypher_verification()