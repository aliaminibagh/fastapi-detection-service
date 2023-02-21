import io

import torch
from PIL import Image


def get_yolov5():
    # local best.pt
    model = torch.hub.load('./app/files/yolov5/', 'custom', path='./app/files/model/fire.pt', source='local')  # local repo
    model.conf = 0.25
    return model


def get_image_from_bytes(binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image
