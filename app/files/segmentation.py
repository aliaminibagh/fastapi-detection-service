import io
import numpy as np
import torch
from PIL import Image
import cv2

def get_yolov5():
    # local best.pt
    model = torch.hub.load('./app/files/yolov5/', 'custom', path='./app/files/model/fire.pt', source='local')  # local repo
    model.conf = 0.25
    return model


def get_image_from_bytes(binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def infer_emotions(image, emotion_model, idx_to_class, test_transforms, device, detector):
    faces = detector.detect_faces(image)
    print(faces)
    print(emotion_model)
    counter = 0
    emotion_list = []
    for bx in faces:
        box = bx['box']

        cropped = image[box[1]:box[1]+box[3],
                        box[0]:box[0] + box[2]]
        cv2.imshow("cropped", cropped)
        cv2.waitKey(0)
        counter += 1

        img_tensor = test_transforms(Image.fromarray(cropped))
        img_tensor.unsqueeze_(0)
        print(img_tensor.shape)
        with torch.no_grad():
            scores = emotion_model(img_tensor.to(device))
        scores = scores[0].data.cpu().numpy()
        emotion = idx_to_class[np.argmax(scores)]
        emotion_list.append(emotion)

    counter = 0
    for em in emotion_list:
        faces[counter]["emotion"] = em
        counter += 1
    return faces
