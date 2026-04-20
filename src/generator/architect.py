import requests, time, os
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class KaggleArchitect:
    def __init__(self, api_token=os.getenv('HF_API_KEY')):
        # self.generator = pipeline("text-generation", model=model_name)

        # We move from a local pipeline to a remote API call
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_token
        )
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"

        # self.api_url = "https://router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        # self.headers = {"Authorization": f"Bearer {api_token}"}


    def generate_strategy(self, query, context_documents):
        #   Prepare the Context
        context_str = "\n---\n".join(context_documents)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        # "content": (
                        #     "You are a Kaggle Grandmaster. "
                        #     "STRICT RULE: Answer ONLY based on the provided context snippets. "
                        #     "If the snippets do not contain the specific model name or architecture details, "
                        #     "state that the information is missing. DO NOT use your general knowledge to guess."
                        #     "If the context snippets are empty or provide no information, you MUST respond with exactly: "
                        #     "'No context provided. I cannot answer the question.' Do not list common models from your training data.")
                        "content": (
                            "You are a Kaggle Grandmaster. Look beyond variable names. "
                            "If you see 'PeftModel' or 'AutoModelForCausalLM', infer the base architecture (e.g., Llama 3, Mistral). "
                            "If you see 'LGBM', explain the ensemble strategy. "
                            "Provide a technical summary of the pipeline, not just a list of names."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context snippets from winning notebooks:\n{context_str}\n\nQuestion: {query}"
                    }
                ],
                max_tokens=500,
                temperature=0.0
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"❌ Architect Error: {str(e)}"
