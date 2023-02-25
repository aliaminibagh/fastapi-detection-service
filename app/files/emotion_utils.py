
import cv2
import numpy as np
import torch
from PIL import Image
import uuid
from .utils_local import get_video_from_bytes

def infer_emotions(image, emotion_model, idx_to_class, test_transforms, device, detector):
    global image_bboxes
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

def get_result_video(binary_video, emotion_model, idx_to_class, test_transforms, device, detector):
    video, _ = get_video_from_bytes(binary_video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # print number of frames in video
    print(f"Number of frames: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")
    # create a video writer
    ID = uuid.uuid4()
    print(ID)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f'./ui/results/{ID}.mp4', fourcc, fps, (width, height))
    count = 0
    while video.isOpened():
        count += 1
        print(100 * "-")
        print(f"Frame {count}")
        ret, frame = video.read()
        if ret:
            _, image = infer_emotions(
                frame, emotion_model, idx_to_class, test_transforms, device, detector)
            out.write(image)
        else:
            break
    video.release()
    out.release()
    # return path of result video
    return f'/results/{ID}.mp4'