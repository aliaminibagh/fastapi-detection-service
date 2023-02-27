import uuid
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from .utils_local import draw_bounding_box_on_image, get_video_from_bytes

class DetectorAPI:
    def __init__(self, path_to_ckpt="./app/files/model/frozen_inference_graph.pb"):
        self.path_to_ckpt = path_to_ckpt
        tf.disable_v2_behavior()
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1]*im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def get_result_video(binary_video, model):
    video, _ = get_video_from_bytes(binary_video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # print number of frames in video
    print(f"Number of frames: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    ID = uuid.uuid4()
    out = cv2.VideoWriter(f"./ui/results/{ID}.mp4", fourcc, fps, (width, height))
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            count += 1
            if count % 10 == 0:
                print(100 * "-")
                print(f"Frame: {count}")
            boxes, scores, classes, num = model.processFrame(frame)
            result = draw_bounding_box_on_image(frame, boxes, scores, classes)
            out.write(result)
            
        else:
            break
    video.release()
    out.release()
    print("video saved to ", f"./ui/results/{ID}.mp4")
    return f'/results/{ID}.mp4'
