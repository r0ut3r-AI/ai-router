from flask import Flask, request, jsonify
import logging
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize OpenAI client for DeepSeek
client = OpenAI(
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com/v1"
 )

# Initialize Flask app
app = Flask(__name__)

# Configure logging to save logs to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_router.log"),  # Simpan log ke file
        logging.StreamHandler()  # Cetak log ke terminal juga
    ]
)

# Function to call DeepSeek API
def call_deepseek_model(input_text):
    start_time = time.time()  # Start timer
    try:
        logging.info(f"Sending request to DeepSeek API with input: {input_text}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": input_text}],
            stream=False
        )
        end_time = time.time()  # End timer
        
        # Debugging: Cetak seluruh response JSON
        response_json = response.model_dump()  # Ubah ke dictionary jika OpenAI SDK pakai pydantic
        logging.info(f"Raw API Response: {response_json}")

        # Pastikan response memiliki struktur yang benar
        if "choices" in response_json and len(response_json["choices"]) > 0:
            response_text = response_json["choices"][0]["message"]["content"]
        else:
            response_text = "No response received from DeepSeek"

        logging.info(f"DeepSeek API Response: {response_text}")
        logging.info(f"Request processed in {end_time - start_time:.2f} seconds")
        return response_text
    except Exception as e:
        logging.error(f"Error calling DeepSeek API: {str(e)}")
        return "Error processing request"

# Available AI models
AI_MODELS = {
    "deepseek": call_deepseek_model
}

@app.route("/route", methods=["POST"])
def route_model():
    start_time = time.time()
    try:
        data = request.json
        model_name = data.get("model")
        input_text = data.get("input")
        client_ip = request.remote_addr
        
        logging.info(f"Incoming request from {client_ip} | Model: {model_name} | Input: {input_text}")
        
        if not model_name or not input_text:
            return jsonify({"error": "Missing required parameters: 'model' or 'input'"}), 400
        
        if model_name not in AI_MODELS:
            return jsonify({"error": "Model not supported"}), 400
        
        # Process input using selected model
        response = AI_MODELS[model_name](input_text)
        
        end_time = time.time()
        logging.info(f"Request completed in {end_time - start_time:.2f} seconds")
        
        return jsonify({"model": model_name, "response": response})
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(debug=True)

