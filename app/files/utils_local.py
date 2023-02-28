from collections import deque
import io
import uuid

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from ultralytics import YOLO

import tempfile


def get_yolov5(name):
    # local best.pt
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=f"./app/files/model/{name}.pt"
    )  # local repo
    model.conf = 0.25
    return model


def get_yolov8(name="yolov8x"):
    # local model
    if name == "smoke":
        model = YOLO("./app/files/model/smoke.pt")
    elif name == "yolov8n":
        model = YOLO("./app/files/model/yolov8n.pt")
    elif name == "yolov8m":
        model = YOLO("./app/files/model/yolov8m.pt")
    else:
        model = YOLO("./app/files/model/yolov8x.pt")
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
                image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2
            )
    return image


def get_video_from_bytes(binary_video):
    try:
        contents = binary_video.file.read()
        with open(binary_video.filename, "wb") as f:
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


def predict_on_live_video(binary_video, model, window_size = 15,classes_list=["non-violence", "violence"]):
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)
    video_reader, _ = get_video_from_bytes(binary_video)
    # Reading the Video File using the VideoCapture Object
    # video_reader = cv2.VideoCapture(binary_video)
    ID  = uuid.uuid4()
    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(f"./ui/results/{ID}.mp4", fourcc, fps, (original_video_width, original_video_height))
    print("Number of Frames in the Video: ", video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:

        # Reading The Frame
        status, frame = video_reader.read()
        frame_id = video_reader.get(1)
        if not status:
            break

        # Resize the Frame to fixed Dimensions

        if frame_id % 100 == 0:
            print(f"Frame {ID} of {video_reader.get(cv2.CAP_PROP_FRAME_COUNT)}")

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized_frame = cv2.resize(rgb_img, (224, 224))
        normalized_frame = resized_frame / 255
        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis=0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(
                predicted_labels_probabilities_deque
            )

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = (
                predicted_labels_probabilities_np.mean(axis=0)
            )

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = (
                0 if predicted_labels_probabilities_averaged[0] < 0.5 else 1
            )

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]

            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(
                frame,
                predicted_class_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Writing The Frame
        video_writer.write(frame)

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    video_writer.release()
    return f'/results/{ID}.mp4'
