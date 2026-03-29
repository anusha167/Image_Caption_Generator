import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import VOCAB_SIZE, MAX_LENGTH, DATA_DIR

def build_tokenizer(captions_dict):
    """Builds a tokenizer from all captions in the dataset"""
    
    # Flatten all captions into one big list
    all_captions = []
    for captions in captions_dict.values():
        for caption in captions:
            all_captions.append(caption)
    
    # Build the tokenizer
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token='<unk>',    # unknown words get this token
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    )
    tokenizer.fit_on_texts(all_captions)
    
    print(f"Vocabulary size: {len(tokenizer.word_index)} unique words")
    print(f"Keeping top {VOCAB_SIZE} words")
    
    return tokenizer

def add_special_tokens(captions_dict):
    """Adds start and end tokens to every caption"""
    
    new_captions = {}
    for image_id, captions in captions_dict.items():
        new_captions[image_id] = []
        for caption in captions:
            caption = '<start> ' + caption + ' <end>'
            new_captions[image_id].append(caption)
    
    return new_captions

def encode_caption(caption, tokenizer):
    """Converts a caption string into a padded sequence of numbers"""
    
    sequence = tokenizer.texts_to_sequences([caption])[0]
    sequence = pad_sequences([sequence], maxlen=MAX_LENGTH, padding='post')
    return sequence