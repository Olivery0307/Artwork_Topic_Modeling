import json
import numpy as np
import umap
import hdbscan
from collections import defaultdict
import os
import glob
import shutil
import yaml

class ImageClusteringPipeline:
    def __init__(self, json_file_path, folder_path):
        self.json_file_path = json_file_path
        self.folder_path = folder_path
        self.art_names = []
        self.embeddings = None
        self.embeddings_data = []
        self.cluster_labels = []

    def load_embeddings(self):
        with open(self.json_file_path, 'r') as json_file:
            self.embeddings_data = json.load(json_file)

        self.art_names = [item['art_name'] for item in self.embeddings_data]
        self.embeddings = np.array([item['embeddings'] for item in self.embeddings_data])
        self.embeddings = self.embeddings.reshape((self.embeddings.shape[0], -1))

        if len(set(self.art_names)) != len(self.art_names):
            unique_data = {}
            for name, emb in zip(self.art_names, self.embeddings):
                unique_data[name] = emb
            self.art_names = list(unique_data.keys())
            self.embeddings = np.array(list(unique_data.values()))

    def cluster_hdbscan_with_umap(self):
        self.load_embeddings()

        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)

        umap_params = config['umap']
        hdbscan_params = config['hdbscan']

        umap_model = umap.UMAP(**umap_params)

        embeddings_dr = umap_model.fit_transform(self.embeddings)

        cluster = hdbscan.HDBSCAN(**hdbscan_params)
        self.cluster_labels = cluster.fit_predict(embeddings_dr)

    def create_cluster_folders(self):
        unique_labels = set(self.cluster_labels)
        for label in unique_labels:
            label_folder = os.path.join(self.folder_path, f'cluster_{label}')
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

    def move_images_to_clusters(self):
        for art_name, cluster_label in zip(self.art_names, self.cluster_labels):
            search_pattern = os.path.join(self.folder_path, '**', art_name + '.*')  # Match any file extension
            image_files = glob.glob(search_pattern, recursive=True)

            if image_files:
                src = image_files[0]  # Take the first match
                dst = os.path.join(self.folder_path, f'cluster_{cluster_label}', os.path.basename(src))
                shutil.move(src, dst)
            else:
                print(f"Image {art_name} not found in {self.folder_path}.")
