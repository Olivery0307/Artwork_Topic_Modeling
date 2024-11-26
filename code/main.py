import os
import yaml
import pandas as pd
from openai import OpenAI

# Import the pipeline classes from their respective modules (assumed to be in the same directory or installed as packages)
from image_embeddings import ImageEmbeddingPipeline
from image_clustering import ImageClusteringPipeline
from description import ImageDescriptionPipeline
from topic_finding import TopicFindingPipeline

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    # Load the configuration settings
    config = load_config(config_path)
    
    # Step 1: Generate Image Embeddings
    print("Step 1: Generating Image Embeddings...")
    embedding_pipeline = ImageEmbeddingPipeline(
        json_file_path=config['paths']['embedding_json'], 
        folder_path=config['paths']['image_folder']
    )
    embedding_pipeline.encode_images()
    embedding_pipeline.save_embeddings()

    # Step 2: Cluster Images based on Embeddings
    print("Step 2: Clustering Images...")
    clustering_pipeline = ImageClusteringPipeline(
        json_file_path=config['paths']['embedding_json'], 
        folder_path=config['paths']['image_folder']
    )
    clustering_pipeline.cluster_hdbscan_with_umap()
    clustering_pipeline.create_cluster_folders()
    clustering_pipeline.move_images_to_clusters()

    # Step 3: Generate Descriptions for Each Cluster
    print("Step 3: Generating Descriptions...")
    description_pipeline = ImageDescriptionPipeline(
        folder_path=config['paths']['image_folder'],
        model_name="YOUR MODEL",
        max_new_tokens=1024
    )
  
    description_pipeline.create_csv(config['paths']['description_csv'])

    # Step 4: Predict Art Genre from Descriptions
    print("Step 4: Finding Topic for each Cluster...")
    topic = TopicFindingPipeline(
        csv_path=config['paths']['description_csv'],
        api_key=config['openai']['api_key']
    )
    prediction = topic.genre_prediction(pd.read_csv(config['paths']['description_csv']))
    print("Topics predicted:\n", prediction)


if __name__ == "__main__":
    config_path = "config.yaml"  
    main(config_path)
