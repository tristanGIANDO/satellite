import torch
from model import UNet

def load_unet_model(path):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
