import os
import random
from PIL import Image
import torch
import csv
from transformers import pipeline, BitsAndBytesConfig

class ImageDescriptionPipeline:
    def __init__(self, folder_path, model_id="YOUR MODEL", max_new_tokens=50):
        self.folder_path = folder_path
        self.max_new_tokens = max_new_tokens
        self.pipe = self.initialize_pipe(model_id)

    def initialize_pipe(self, model_id):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        return pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

    def load_random_images(self, folder_path, num_images=5):
        images = []
        image_files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

        if len(image_files) < num_images:
            print(f"Only found {len(image_files)} images in the folder. Loading all of them.")
            num_images = len(image_files)

        random.shuffle(image_files)
        selected_files = image_files[:num_images]

        for filename in selected_files:
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    images.append((img.copy(), filename))
            except Exception as e:
                print(f"Failed to load image {filename}: {e}")

        return images

    def prepare_folders(self):
        """
        This function iterates over sub-folders within the main folder,
        preparing images in each sub-folder and storing them in a dictionary.
        """
        folders_images = {}

        for folder_name in os.listdir(self.folder_path):
            sub_folder_path = os.path.join(self.folder_path, folder_name)
            if os.path.isdir(sub_folder_path):
                folders_images[folder_name] = self.load_random_images(sub_folder_path)

        return folders_images

    def generate_description(self, image):
        prompt = f"""
        USER: <image>
        Provide detailed description of the given artwork.
        ASSISTANT:
        """
        outputs = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": self.max_new_tokens, "repetition_penalty": 1.2})
        result = outputs[0]["generated_text"].split("ASSISTANT:")[1:][0]

        return result

    def extract_cluster_number(self, cluster_name):
        """
        Extracts the numeric part from the cluster name (e.g., "cluster_2" -> "2").
        """
        return ''.join(filter(str.isdigit, cluster_name))

    def create_csv(self, csv_filename):
        """
        This function creates a CSV file with image descriptions.
        """
        folders_images = self.prepare_folders()

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["art_name", "cluster", "description"])

            for cluster, images in folders_images.items():
                cluster_number = self.extract_cluster_number(cluster)
                for image, filename in images:
                    art_name = os.path.splitext(filename)[0]
                    with torch.no_grad():
                        description = self.generate_description(image)

                    torch.cuda.empty_cache()
                    writer.writerow([art_name, cluster_number, description])

        print(f"CSV file created at: {csv_filename}")

# Example usage:
# pipeline = ImageDescriptionPipeline(folder_path="/path/to/folder")
# pipeline.create_csv("output.csv")
