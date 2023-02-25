import uuid
import torch
import numpy as np
import cv2
from time import time

class OD:
    def __init__(self, capture_index, model):
        self.capture_index = capture_index
        # self.model = self.load_model(model_name)
        self.model = model
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
        # if model_name:
        #     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        # else:
        #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # return model
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=f'./app/files/model/{model_name}.pt')
        model.conf = 0.25
        return model
    

    def score_frame(self, frame):
        """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
        # return results

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self,labels, cord, frame):
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
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

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
        print(ID)
        out = cv2.VideoWriter(f'./ui/results/{ID}.mp4', fourcc, fps, (width, height))
        print(f"Number of frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        count = 0
        while cap.isOpened():
            count += 1
            print(100 * "-")
            print(f"Frame {count}")
        
            ret, frame = cap.read()
    
            
            if ret:
                # frame = cv2.resize(frame, (640,640))
                labels, cord= self.score_frame(frame)
                frame = self.plot_boxes(labels, cord, frame)
                out.write(frame)
            #print(f"Frames Per Second : {fps}")
            
            # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            # cv2.imshow('YOLOv5 Detection', frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return f'/results/{ID}.mp4'