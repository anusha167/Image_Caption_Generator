import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMAGE_SIZE = (299, 299) # InceptionV3 input size
CHANNELS = 3

EMBEDDING_DIM = 256
UNITS = 512
VOCAB_SIZE = 5000
MAX_LENGTH = 50

BATCH_SIZE = 64
EPOCHS = 20 # Epoch is the number of times the entire training dataset is passed through the model
LEARNING_RATE = 0.001