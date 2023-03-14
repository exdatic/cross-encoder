from sentence_transformers import CrossEncoder
import torch


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    # for pinging the model
    if not model_inputs:
        return {}
    # Parse out your arguments
    sentences = model_inputs.get('sentence_pairs', None)
    if sentences == None:
        return {'message': "No sentences provided"}
    
    # Run the model
    result = model.predict(sentences)

    return {"output": result.tolist()}
