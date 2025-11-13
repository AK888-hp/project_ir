import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import io

def load_model():
    """
    Loads the VGG16 model, pre-trained on ImageNet.
    We are using the output of the 'fc2' layer as our feature vector.
    """
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    return model

def extract_features_from_path(img_path, model):
    """
    Extracts features from an image file on disk.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    return features.flatten() # Return a flat 1D vector

def extract_features_from_bytes(img_bytes, model):
    """
    Extracts features from an image sent as bytes (e.g., from an API upload).
    """
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    return features.flatten() # Return a flat 1D vector