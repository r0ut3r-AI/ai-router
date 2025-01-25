import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Muat variabel dari .env

class GPTModel:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Contoh penggunaan
if __name__ == "__main__":
    gpt = GPTModel()
    response = gpt.generate("Jelaskan AI dalam satu kalimat.")
    print("GPT-3.5:", response)
