# In this file, we define download_model
# It runs during container build time to get model weights built into the container
from sentence_transformers import CrossEncoder
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

if __name__ == "__main__":
    download_model()
