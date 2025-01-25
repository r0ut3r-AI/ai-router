from flask import Flask, jsonify, request
from distilbert_model import DistilBERTModel

app = Flask(__name__)
distilbert_model = DistilBERTModel()

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Parameter 'text' diperlukan"}), 400

    try:
        result = distilbert_model.predict(text)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
