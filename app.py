from flask import Flask, jsonify, request
from gpt_model import GPTModel
from llama_model import LlamaModel

app = Flask(__name__)
gpt_model = GPTModel()
llama_model = LlamaModel()

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    prompt = data.get("prompt")
    model = data.get("model", "gpt")  # Default: GPT-3.5

    if not prompt:
        return jsonify({"error": "Parameter 'prompt' diperlukan"}), 400

    try:
        if model == "gpt":
            response = gpt_model.generate(prompt)
        elif model == "llama":
            response = llama_model.generate(prompt)
        else:
            return jsonify({"error": "Model tidak valid"}), 400

        return jsonify({"response": response, "model": model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
