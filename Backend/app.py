from flask import Flask, request, jsonify, send_file
from predict import detect_forgery
from history_plot import get_training_plot
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    image.save(path)

    output_path, prediction = detect_forgery(path)
    return send_file(output_path, mimetype='image/jpeg')

@app.route('/stats', methods=['GET'])
def stats():
    return get_training_plot()

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/metrics', methods=['GET'])
def metrics():
    metrics_path = os.path.join("model_output", "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not found. Please train the model."}), 404

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    return jsonify(metrics_data)
