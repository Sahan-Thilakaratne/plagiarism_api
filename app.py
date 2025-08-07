from flask import Flask, request, jsonify
from paraphase_detector import compute_similarity, classify_similarity
from plagarism_detector import get_bert_embeddings
import joblib
import os

app = Flask(__name__)

# Load AI/Human classifier
model_path = "bert_ai_text_detector.pkl"
if os.path.exists(model_path):
    classifier = joblib.load(model_path)
else:
    classifier = None


@app.route("/detect-paraphrase", methods=["POST"])
def detect_paraphrase():
    data = request.json
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")
    similarity = compute_similarity(text1, text2)
    classification = classify_similarity(similarity)
    return jsonify({
        "similarity_score": round(similarity, 4),
        "classification": classification
    })


@app.route("/detect-ai", methods=["POST"])
def detect_ai():
    if not classifier:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    text = data.get("text", "")
    embedding = get_bert_embeddings(text).reshape(1, -1)
    prediction = classifier.predict(embedding)[0]
    label = ["human", "ai"][prediction]
    return jsonify({"prediction": label})


@app.route("/")
def index():
    return "Plagiarism & Paraphrase Detection API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
