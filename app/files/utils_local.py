import io

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import tempfile


def get_yolov5(name):
    # local best.pt
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=f'./app/files/model/{name}.pt')  # local repo
    model.conf = 0.25
    return model


def get_yolov8(name="yolov8x"):
    # local model
    if name == "mokes":
        model = YOLO('./app/files/model/smoke.pt')
    elif name == "yolov8n":
        model = YOLO('./app/files/model/yolov8n.pt')
    elif name == "yolov8m":
        model = YOLO('./app/files/model/yolov8m.pt')
    else:
        model = YOLO('./app/files/model/yolov8x.pt')
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


def get_video_from_bytes(binary_video):
    try:
        contents = binary_video.file.read()
        with open(binary_video.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        binary_video.file.close()
    return cv2.VideoCapture(binary_video.filename), binary_video.filename

def get_video_from_bytes_temp(binary_video):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(binary_video.file.read())
        return cv2.VideoCapture(tmp.name), tmp.name




