from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import load_model, extract_features_from_bytes # Import from our other file

app = Flask(__name__)
CORS(app) # Allows your frontend to talk to this backend

# --- Load Model and Index ONCE when the server starts ---
print("Loading model...")
model = load_model()
print("Model loaded.")

print("Loading feature index...")
with open('features_index.pkl', 'rb') as f:
    all_features = pickle.load(f)
# Separate filenames and vectors for faster searching
image_paths = list(all_features.keys())
feature_vectors = np.array(list(all_features.values()))
print(f"Index loaded for {len(image_paths)} images.")
# -----------------------------------------------------

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image file in-memory
        img_bytes = file.read()
        
        # 1. Get features for the query image
        query_features = extract_features_from_bytes(img_bytes, model)
        query_features = query_features.reshape(1, -1) # Reshape for cosine_similarity

        # 2. Calculate similarity against all indexed images
        similarities = cosine_similarity(query_features, feature_vectors)
        
        # 3. Get top 10 results (indices)
        top_indices = similarities[0].argsort()[-10:][::-1]
        
        # 4. Get the corresponding image paths
        results = [image_paths[i] for i in top_indices]
        
        # 5. Return results as JSON
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the AI service on port 5000
    print("Starting Flask server... Access it at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)