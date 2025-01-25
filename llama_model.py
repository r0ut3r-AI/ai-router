import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class LlamaModel:
    def __init__(self):
        self.client = InferenceClient(token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model = "meta-llama/Llama-2-7b-chat-hf"  # Model Llama 2

    def generate(self, prompt):
        response = self.client.post(
            json={"inputs": prompt},
            model=self.model
        )
        return response.json()[0]['generated_text']

# Contoh penggunaan
if __name__ == "__main__":
    llama = LlamaModel()
    response = llama.generate("Apa ibukota Indonesia?")
    print("Llama 2:", response)
