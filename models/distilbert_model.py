import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

class DistilBERTModel:
    def __init__(self):
        # Pilih task sesuai kebutuhan (contoh: text-classification)
        self.task = "text-classification"
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Contoh untuk analisis sentimen
        self.model = pipeline(
            self.task,
            model=self.model_name,
        )

    def predict(self, text):
        result = self.model(text)
        return result

# Contoh penggunaan
if __name__ == "__main__":
    model = DistilBERTModel()
    response = model.predict("I love this project!")
    print(response)
