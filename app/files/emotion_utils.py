
import cv2
import numpy as np
import torch
from PIL import Image


def infer_emotions(image, emotion_model, idx_to_class, test_transforms, device, detector):
    faces = detector.detect_faces(image)
    counter = 0
    emotion_list = []
    for bx in faces:
        box = bx['box']

        cropped = image[box[1]:box[1]+box[3],
                        box[0]:box[0] + box[2]]
        counter += 1

        img_tensor = test_transforms(Image.fromarray(cropped))
        img_tensor.unsqueeze_(0)
        with torch.no_grad():
            scores = emotion_model(img_tensor.to(device))
        scores = scores[0].data.cpu().numpy()
        emotion = idx_to_class[np.argmax(scores)]
        emotion_list.append(emotion)
        image_bboxes = cv2.rectangle(
            image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        cv2.putText(image_bboxes, emotion,
                    (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    counter = 0
    for em in emotion_list:
        faces[counter]["emotion"] = em
        counter += 1

    return faces, image_bboxes