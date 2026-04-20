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
                        "content": (
                            "You are a Kaggle Grandmaster. "
                            "STRICT RULE: Answer ONLY based on the provided context snippets. "
                            "If the snippets do not contain the specific model name or architecture details, "
                            "state that the information is missing. DO NOT use your general knowledge to guess."
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

        #   The Prompt (Instruction Tuning)
        # prompt = f"""
        #         You are a Kaggle Grandmaster. Use the following snippets from a winning notebook to answer the user's question.
        #
        #         If the information is not in the context, strictly state that the notebook does not contain this information.
        #
        #         CONTEXT FROM WINNING NOTEBOOK:
        #         {context_str}
        #
        #         USER QUESTION: {query}
        #
        #         GRANDMASTER RESPONSE:
        #         """

        # We use a structured prompt that Instruct models understand
        # prompt = f"<s>[INST] Use these Kaggle snippets to answer: {query}\n\nSnippets:\n{context_str} [/INST]</s>"
        #
        # payload = {
        #     "inputs": prompt,
        #     "parameters": {
        #         "max_new_tokens": 500,
        #         "temperature": 0.1,
        #         "return_full_text": False
        #     },
        #     "options": {"wait_for_model": True}  # CRITICAL: Tells HF to load the model if it's "sleeping"
        # }
        #
        # try:
        #     response = requests.post(self.api_url, headers=self.headers, json=payload)
        #
        #     if response.status_code == 503:
        #         print("⏳ Model is loading on Hugging Face... waiting 10s")
        #         time.sleep(10)
        #         response = requests.post(self.api_url, headers=self.headers, json=payload)
        #
        #     response.raise_for_status()
        #     return response.json()[0]['generated_text'].strip()
        #
        # except Exception as e:
        #     return f"❌ Error: {str(e)} | Response: {response.text}"
            # We add 'return_full_text=False' to stop the mirror effect
        # response = self.generator(
        #     prompt,
        #     max_new_tokens=100,
        #     truncation=True,
        #     return_full_text=False,  # THIS IS THE KEY
        #     pad_token_id=50256  # Avoids warning for GPT2
        # )
        #
        # answer = response[0]['generated_text'].strip()
        # return answer if answer else "The model could not generate a reasoning for this context."

