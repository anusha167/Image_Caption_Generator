import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from config import MODELS_DIR, MAX_LENGTH
from feature_extractor import build_feature_extractor, preprocess_image

def generate_caption(image_path, model, tokenizer, feature_extractor):
    """Given an image, generate a caption"""
    
    # Extract features from the image
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    feature = feature_extractor.predict(img, verbose=0)
    
    # Start with the <start> token
    caption = '<start>'
    
    for _ in range(MAX_LENGTH):
        # Encode the current caption
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence], maxlen=MAX_LENGTH, padding='post'
        )
        
        # Predict the next word
        predictions = model.predict([feature, sequence], verbose=0)
        next_word_idx = np.argmax(predictions)
        
        # Convert index back to word
        next_word = tokenizer.index_word.get(next_word_idx, None)
        
        if next_word is None or next_word == '<end>':
            break
            
        caption += ' ' + next_word
    
    # Clean up the caption
    caption = caption.replace('<start>', '').strip()
    return caption

def main():
    print("Loading model and tokenizer...")
    model = tf.keras.models.load_model(f"{MODELS_DIR}/best_model.keras")
    
    with open(f"{MODELS_DIR}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    feature_extractor = build_feature_extractor()
    
    # Test on a val image
    import random
    from pathlib import Path
    from config import DATA_DIR
    
    images = list(Path(f"{DATA_DIR}/val2017").glob("*.jpg"))
    test_image = random.choice(images)
    
    print(f"Image: {test_image.name}")
    caption = generate_caption(test_image, model, tokenizer, feature_extractor)
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main()