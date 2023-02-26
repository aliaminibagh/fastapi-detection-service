import uuid
from time import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class OD():
    def __init__(self, capture_index, model_name, yolo_version="five"):
        self.capture_index = capture_index
        self.yolo_version = yolo_version
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
        if model_name == "fire":
            model = torch.hub.load('ultralytics/yolov5',
                                   'custom', path=f'./app/files/model/fire.pt')
            self.yolo_version = "five"
        elif model_name == "knife":
            model = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=f'./app/files/model/knife.pt')
            self.yolo_version = "five"
        elif model_name == "arms":
            model = torch.hub.load('ultralytics/yolov5',
                                   'custom', path=f'./app/files/model/arms.pt')
            self.yolo_version = "five"
        elif model_name == "smoke":
            model = YOLO('./app/files/model/smoke.pt')
            self.yolo_version = "eight"
        elif model_name == "yolov8n":
            model = YOLO('./app/files/model/yolov8n.pt')
            self.yolo_version = "eight"
        elif model_name == "yolov8m":
            model = YOLO('./app/files/model/yolov8m.pt')
            self.yolo_version = "eight"
        else:
            model = YOLO('./app/files/model/yolov8x.pt')
            self.yolo_version = "eight"
        return model

    def score_frame(self, frame):
        """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = frame
        results = self.model(frame)
        if self.yolo_version == "five":
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        else:
            labels, cord = results[0].boxes.cls, torch.cat(
                [results[0].boxes.xyxyn, results[0].boxes.conf.reshape(-1, 1)], axis=1)
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, labels, cord, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        # labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.25:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(
                    labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ID = uuid.uuid4()
        out = cv2.VideoWriter(
            f'./ui/results/{ID}.mp4', fourcc, fps, (width, height))
        print(f"Number of frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        count = 0
        while cap.isOpened():
            count += 1
            if count % 100 == 0:
                print(100 * "-")
                print(f"Frame {count}")
            ret, frame = cap.read()

            if ret:
                # frame = cv2.resize(frame, (640,640))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                labels, cord = self.score_frame(frame)
                frame = self.plot_boxes(labels, cord, frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("video saved to: ", f'./ui/results/{ID}.mp4')
        return f'/results/{ID}.mp4'
