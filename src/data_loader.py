import os
import json
from pathlib import Path
from config import DATA_DIR

def load_captions():
    """Loads captions from the annotation file into a dictionary"""
    
    captions_file = os.path.join(DATA_DIR, "annotations", "captions_val2017.json")    
    
    with open(captions_file, "r") as f:
        data = json.load(f)
    
    # Build a dictionary: {image_id: [caption1, caption2, ...]}
    captions_dict = {}
    
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        
        if image_id not in captions_dict:
            captions_dict[image_id] = []
        
        captions_dict[image_id].append(caption)
    
    print(f"Loaded captions for {len(captions_dict)} images")
    return captions_dict

def load_image_paths():
    """Gets all image paths from the val2017 folder"""
    
    images_dir = Path(DATA_DIR) / "val2017"
    image_paths = list(images_dir.glob("*.jpg"))
    
    print(f"Found {len(image_paths)} images")
    return image_paths