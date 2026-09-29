from flask import Flask, request, jsonify
from flask_cors import CORS
from classifier import classify

app = Flask(__name__, static_folder="static")
CORS(app)


@app.get("/")
def home():
    """Serve the frontend."""
    with open("static/index.html", encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}


@app.post("/classify")
def api_classify():
    """
    Accepts: { "url": "http://..." }
    Returns: { url, level, score, probabilities, explanation, features }
    """
    data = request.get_json(silent=True) or {}
    url  = str(data.get("url", "")).strip()

    if not url:
        return jsonify({"error": "Please provide a URL."}), 400

    try:
        return jsonify(classify(url))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"})
