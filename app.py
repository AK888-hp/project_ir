import os
import base64
import mimetypes
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename # --- 1. ADD THIS IMPORT ---
from feature_extractor import load_model, extract_features_from_bytes #
from sklearn.metrics.pairwise import cosine_similarity

# --- Auto-detect the dataset directory ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIRECTORY = os.path.join(BASE_DIR, 'dataset')
INDEX_FILE = 'features_index.pkl'
# ----------------------------------------

app = Flask(__name__)
CORS(app) #

# --- Load Model and Index ---
print("Loading model...")
model = load_model()
print("Model loaded.")

print("Loading feature index...")
with open(INDEX_FILE, 'rb') as f:
    all_features = pickle.load(f)
image_paths = list(all_features.keys())
feature_vectors = np.array(list(all_features.values()))
print(f"Index loaded for {len(image_paths)} images.")
# -----------------------------------------------------

def encode_image_to_base64(image_path):
    """Reads an image file and returns its Base64 data URL."""
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None: mime_type = 'application/octet-stream'
        with open(image_path, 'rb') as f:
            image_data = f.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

# --- 2. ADD THIS ENTIRE NEW ROUTE ---
@app.route('/upload_image', methods=['POST'])
def upload_image():
    global all_features, image_paths, feature_vectors # We need to update the global variables
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        # Create the full, absolute path to save the file
        save_path = os.path.join(DATASET_DIRECTORY, filename)
        
        # Read the file bytes into memory
        img_bytes = file.read()
        
        # 1. Extract features from the bytes
        new_features = extract_features_from_bytes(img_bytes, model)
        
        # 2. Save the image file to the dataset folder
        with open(save_path, 'wb') as f:
            f.write(img_bytes)
        
        # 3. Update the features_index.pkl file
        #    We must load, update, and re-save the whole file
        with open(INDEX_FILE, 'rb') as f:
            all_features_disk = pickle.load(f)
            
        all_features_disk[save_path] = new_features # Add the new entry
        
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump(all_features_disk, f) # Re-save the file
            
        # 4. Update the in-memory variables for the search
        #    (This avoids needing to restart the server)
        all_features = all_features_disk
        image_paths = list(all_features.keys())
        feature_vectors = np.array(list(all_features.values()))
        
        print(f"--- Image added: {filename} ---")
        print(f"Index updated. Total images: {len(image_paths)}")
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'message': f'Image added and indexed successfully. Total images: {len(image_paths)}'
        })

    except Exception as e:
        print(f"Error uploading image: {e}")
        return jsonify({'error': str(e)}), 500
# ---------------------------------------

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
        
        top_paths = [image_paths[i] for i in top_indices]
        results_as_base64 = [encode_image_to_base64(p) for p in top_paths]
        valid_results = [r for r in results_as_base64 if r is not None]
        
        return jsonify({'results': valid_results})

    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({'error': str(e)}), 500
# -----------------------------------------------------

if __name__ == '__main__':
    print("Starting Flask server... Access it at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)