import os
import pickle
from feature_extractor import load_model, extract_features_from_path #

# --- NEW: Auto-detect the dataset folder ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
YOUR_DATASET_FOLDER_PATH = os.path.join(BASE_DIR, 'dataset')
print(f"--- Indexing images from: {YOUR_DATASET_FOLDER_PATH} ---")
# ---------------------------------------------------

dataset_path_norm = os.path.normpath(YOUR_DATASET_FOLDER_PATH)

model = load_model()
all_features = {}

print("Starting feature extraction...")

for root, dirs, files in os.walk(dataset_path_norm):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            full_path_norm = os.path.normpath(full_path) # Get the full absolute path
            
            try:
                features = extract_features_from_path(full_path_norm, model)
                
                # --- CRITICAL CHANGE ---
                # We are now saving the FULL ABSOLUTE PATH as the key
                all_features[full_path_norm] = features
                # -----------------------
                
                print(f"Processed: {full_path_norm}")
                
            except Exception as e:
                print(f"Error processing {full_path}: {e}")

# Save the entire dictionary of features to a pickle file
with open('features_index.pkl', 'wb') as f:
    pickle.dump(all_features, f)

print(f"\nFeature extraction complete! Index saved to 'features_index.pkl'.")
print(f"Indexed {len(all_features)} images.")