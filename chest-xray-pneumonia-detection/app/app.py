from flask import Flask, request, jsonify

app = Flask(__name__)

# Placeholder - in production load a real model and preprocessing pipeline
MODEL_PATH = "../models/best_model.h5"

@app.route('/')
def index():
    return jsonify({"message": "Chest X-Ray Pneumonia Detection API (placeholder)"})

@app.route('/predict', methods=['POST'])
def predict():
    # Expecting a JSON payload with a base64-encoded image or a path; this is a placeholder
    data = request.get_json(silent=True) or {}
    return jsonify({
        "prediction": "placeholder",
        "confidence": 0.0,
        "received": bool(data)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
