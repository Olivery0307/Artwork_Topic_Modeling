import pandas as pd
import numpy as np
from openai import OpenAI
import os
import yaml


class TopicFindingPipeline:
    def __init__(self, csv_path, config_path):

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.api_key = config['openai']['api_key']
        self.df = pd.read_csv(csv_path)
        self.client = OpenAI(api_key=self.api_key)

    def commonality_together(self, df):
        df_string = df.to_string(index=False)

        prompt = f"""
        USER:
        YOUR PROMPT

        ASSISTANT:
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
        )
        response = chat_completion.choices[0].message.content
        return response
    
    def visual_feldman(self,df):
        """
        This function generates a visual analysis of artworks from various clusters using Feldman's method of art critique.
        """
        df_string = df.to_string(index=False)

        prompt = f"""
        USER:
        YOUR PROMPT

        ASSISTANT:
        """
        chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-4",
            )
        response = chat_completion.choices[0].message.content
        return response
    
    def content_topic(self,df):
        """
        This function generates a content analysis of artworks from various clusters using Feldman's method of art critique.
        """
        df_string = df.to_string(index=False)

        prompt = f"""
        USER:
        YOUR PROMPT

        ASSISTANT:
        """
        chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-4",
            )
        response = chat_completion.choices[0].message.content
        return response
    
    


    
