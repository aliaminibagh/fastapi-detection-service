import io

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


def get_yolov5(name):
    # local best.pt
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=f'./app/files/model/{name}.pt')  # local repo
    model.conf = 0.25
    return model


def get_yolov8():
    model = YOLO('./app/files/model/smoke.pt')
    return model


def get_image_from_bytes(binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_image_with_cv2(binary_image):
    nparr = np.frombuffer(binary_image, np.uint8)
    input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.resize(input_image, (1280, 720), interpolation=cv2.INTER_AREA)
    return input_image


def draw_bounding_box_on_image(image, boxes, scores, classes, threshold=0.7):
    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            image = cv2.rectangle(
                image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
    return image
