import os
import pickle
from feature_extractor import load_model, extract_features_from_path # Import from our other file

# --- IMPORTANT: CHANGE THIS ---
# Point this to the folder containing all your medical images
YOUR_DATASET_FOLDER_PATH = "C:/Users/Anantha krishna rao/OneDrive/Desktop/ai-service/dataset"
# ------------------------------

model = load_model()
all_features = {}
image_paths = []

print("Starting feature extraction...")

# Recursively find all images (jpg, png) in the dataset folder
for root, dirs, files in os.walk(YOUR_DATASET_FOLDER_PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)

# Now extract features for each image
for img_path in image_paths:
    try:
        features = extract_features_from_path(img_path, model)
        # We store the image path (relative to the dataset root) and its features
        # This assumes your frontend can access images based on this path
        relative_path = os.path.relpath(img_path, YOUR_DATASET_FOLDER_PATH) 
        all_features[relative_path] = features
        print(f"Processed: {relative_path}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Save the entire dictionary of features to a pickle file
with open('features_index.pkl', 'wb') as f:
    pickle.dump(all_features, f)

print(f"\nFeature extraction complete! Index saved to 'features_index.pkl'.")
print(f"Indexed {len(all_features)} images.")
