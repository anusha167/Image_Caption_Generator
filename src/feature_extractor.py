import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
from config import IMAGE_SIZE

def build_feature_extractor():
    """Builds the InceptionV3 model for feature extraction"""

    inception = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    
    model = Model(inputs=inception.input, outputs=inception.output)
    
    return model

def preprocess_image(image_path):
    """Preprocesses the image for feature extraction"""
    
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img = np.array(img)
    img = preprocess_input(img)

    return img

def extract_features(image_paths, model):
    """Extracts features from a list of image paths using the given model"""
    
    features = {}
    
    for i, path in enumerate(image_paths):
        img = preprocess_image(path)
        img = np.expand_dims(img, axis=0)
        
        feature = model.predict(img, verbose=0)
        image_id = path.stem
        features[image_id] = feature

        if i % 100 == 0:
            print(f"Extracted features from {i}/{len(image_paths)} images")

    return features