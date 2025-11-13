import os
import pickle
from feature_extractor import load_model, extract_features_from_path #

# --- IMPORTANT: Point this to your dataset folder ---
YOUR_DATASET_FOLDER_PATH = "C:/Users/Anantha krishna rao/OneDrive/Desktop/ai-service/dataset"
# ---------------------------------------------------

# --- NEW: Normalize the dataset path ---
# This fixes any mix of forward/backward slashes
dataset_path_norm = os.path.normpath(YOUR_DATASET_FOLDER_PATH)
# -------------------------------------

model = load_model()
all_features = {}
image_paths = []

print("Starting feature extraction...")

# Recursively find all images (jpg, png) in the dataset folder
for root, dirs, files in os.walk(dataset_path_norm):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            
            # --- NEW: Normalize the full image path ---
            full_path_norm = os.path.normpath(full_path)
            # ----------------------------------------
            
            try:
                features = extract_features_from_path(full_path_norm, model)
                
                # --- NEW: Create a clean relative path ---
                relative_path = os.path.relpath(full_path_norm, dataset_path_norm) 
                # CRITICAL: Replace Windows backslashes with web-safe forward slashes
                relative_path_web = relative_path.replace(os.path.sep, '/')
                # -----------------------------------------
                
                all_features[relative_path_web] = features
                print(f"Processed: {relative_path_web}")
                
            except Exception as e:
                print(f"Error processing {full_path}: {e}")

# Save the entire dictionary of features to a pickle file
with open('features_index.pkl', 'wb') as f:
    pickle.dump(all_features, f)

print(f"\nFeature extraction complete! Index saved to 'features_index.pkl'.")
print(f"Indexed {len(all_features)} images.")