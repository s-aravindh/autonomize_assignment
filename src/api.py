from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import json

from model import CpGPredictor

def load_model(model_path, config_path):
    """
    model: load trained pytorch model
    config: load config specific to that trained model
    """
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    model = CpGPredictor(num_classes=config["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, config

#  load 128 dimension model with no padding
fixed_dim_model, fixed_dim_config = load_model(
    '../trained_models_and_config/128dim_model.pt',
    "../trained_models_and_config/128_fixed_dim_config.json"
    )
#  load 128 dimension model with padding
padded_dim_model, padded_dim_config = load_model(
    '../trained_models_and_config/128dim_padding_model.pt',
    "../trained_models_and_config/128_padded_dim_config.json"
    )

app = FastAPI()


class InputData(BaseModel):
    model_type: str
    dna_char: str

def preprocess(model_type:str, dna_char:str):
    if model_type == "fixed":
        config = fixed_dim_config
        model = fixed_dim_model
        dna2int = config["dna2int"]
        dna_list = [dna2int.get(i) for i in dna_char]
        data = torch.Tensor(dna_list)
    if model_type == "padded":
        config = padded_dim_config
        model = padded_dim_model
        dna2int = config["dna2int"]
        dna_list = [dna2int.get(i) for i in dna_char]
        data = torch.Tensor(dna_list)
        data = nn.ConstantPad1d((0, 128 - data.shape[0]), 0)(data)
    return config, model, data

@app.post("/predict")
async def predict(data: InputData):
    """
    takes dna list as input and asks
    "model_type": "fixed/padded",
    "dna_char": "GGGNCGCCCNCTCTTAGGGGGAANNCATTTNGACTGNGTNCGTNTGCAAATACTGNANNTGCCGTGTAATTATNNCGNTACTGTTNNGCNCCACNGCCCAGNAGNTGAGNG"
    """
    try:
        # Convert the input data to a Torch tensor

        config, model, input = preprocess(data.model_type, data.dna_char)

        # Make a prediction
        with torch.no_grad():
            probabilities = model(input.unsqueeze(0))

        # Get the predicted class
        predicted_class = torch.argmax(probabilities).item()

        return {"number of CpGs": config["classes"][predicted_class]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
