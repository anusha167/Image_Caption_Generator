import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, 
    Dropout, Add
)
from config import EMBEDDING_DIM, UNITS, VOCAB_SIZE, MAX_LENGTH

def build_caption_model():
    """Builds the Show and Tell caption generation model"""
    
    image_input = Input(shape=(2048,), name='image_input')
    image_dense = Dense(EMBEDDING_DIM, activation='relu')(image_input)
 
    caption_input = Input(shape=(MAX_LENGTH,), name='caption_input')
    caption_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(caption_input)
    caption_lstm = LSTM(UNITS)(caption_embedding)
    caption_dense = Dense(EMBEDDING_DIM, activation='relu')(caption_lstm)
   
    merged = Add()([image_dense, caption_dense])
    output = Dense(VOCAB_SIZE, activation='softmax')(merged)
  
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    return model