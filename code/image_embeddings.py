import torch
from PIL import Image
import open_clip
import glob
import json
import os
from torchvision import transforms
import time

class ImageEmbeddingPipeline:
    def __init__(self, model_name='YOUR MODEL', pretrained='YOUR PRETRAINED MODEL', folder_path="../Images"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval() 
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.folder_path = folder_path
        self.image_paths = self._get_image_paths()
        self.embeddings_data = []
        
    def _get_image_paths(self):
        image_paths = glob.glob(f"{self.folder_path}/**/*.jpg", recursive=True) + \
                      glob.glob(f"{self.folder_path}/**/*.jpeg", recursive=True) + \
                      glob.glob(f"{self.folder_path}/**/*.png", recursive=True)
        return image_paths

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0)
        return image

    def encode_images(self):
        with torch.no_grad(), torch.cuda.amp.autocast():
            for image_path in self.image_paths:
                image = self.preprocess_image(image_path)
                image_embeddings = self.model.encode_image(image)
                art_name = os.path.splitext(os.path.basename(image_path))[0]
                self.embeddings_data.append({
                    "art_name": art_name,
                    "embeddings": image_embeddings.tolist()
                })

    def save_embeddings(self, json_file_path='../data/embeddings.json'):
        with open(json_file_path, 'w') as json_file:
            json.dump(self.embeddings_data, json_file, indent=4)
        print(f"Embeddings saved to {json_file_path}")

        