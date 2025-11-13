import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import load_model, extract_features_from_bytes #

# Automatically find the dataset directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIRECTORY = os.path.join(BASE_DIR, 'dataset') #

app = Flask(__name__)
CORS(app) #

# --- Load Model and Index ---
print("Loading model...")
model = load_model()
print("Model loaded.")
print("Loading feature index...")
with open('features_index.pkl', 'rb') as f:
    all_features = pickle.load(f)
image_paths = list(all_features.keys())
feature_vectors = np.array(list(all_features.values()))
print(f"Index loaded for {len(image_paths)} images.")
print(f"--- Serving images from: {DATASET_DIRECTORY} ---") # Check this path
# -----------------------------------------------------

# --- This is the route that serves the images ---
@app.route('/static/<path:filename>')
def serve_static_image(filename):
    
    # --- THIS IS THE CRITICAL DEBUGGING CHECK ---
    full_path = os.path.join(DATASET_DIRECTORY, filename)
    print(f"--- Attempting to serve: {full_path} ---")
    if not os.path.exists(full_path):
        print(f"--- ERROR: File does not exist at path: {full_path} ---")
    else:
        print(f"--- SUCCESS: File exists. Sending... ---")
    # -----------------------------------------------
    
    return send_from_directory(DATASET_DIRECTORY, filename)
# -----------------------------------------------------

@app.route('/search', methods=['POST'])
def search(): #
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        query_features = extract_features_from_bytes(img_bytes, model)
        query_features = query_features.reshape(1, -1) 
        similarities = cosine_similarity(query_features, feature_vectors)
        top_indices = similarities[0].argsort()[-10:][::-1]
        results = [image_paths[i] for i in top_indices]
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# -----------------------------------------------------

if __name__ == '__main__':
    print("Starting Flask server... Access it at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)