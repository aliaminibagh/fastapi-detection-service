
import uuid

import cv2 as cv
import torch
from fastapi import APIRouter, File, UploadFile
from mtcnn_cv2 import MTCNN
from torchvision import transforms

from ..files.emotion_utils import infer_emotions, get_result_video
from ..files.utils_local import get_image_with_cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_model = torch.load('./app/files/model/EmotionNet_b27.pt')

if torch.cuda.is_available():
    emotion_model.cuda()
emotion_model.eval()
idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness',
                4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
test_transforms = transforms.Compose(
    [
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)

detector = MTCNN()


router = APIRouter(
    prefix="/emotions",
    tags=["emotions"],
)


@router.post("/infer-image", summary='Detect emotions of faces in image and return json', response_description="Something here")
async def emotion_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    results, image = infer_emotions(
        input_image, emotion_model, idx_to_class, test_transforms, device, detector)
    if len(results) == 0:
        return {"result": "چهره ای در تصویر برای عملیات تشخیص احساسات یافت نشده است"}
    else:
        ID = uuid.uuid4()
        cv.imwrite(f"./ui/results/{ID}.jpg", image)
    # return ({f'Face_{num+1}': i['box'], 'Confidence': round(i['confidence'], 4), 'Emotion': i['emotion']} for num, i in enumerate(results))
        return {"result": [{'x_min': int(box[0]), 'y_min': int(box[1]), 'x_max': int(box[0] + box[2]), 'y_max': int(box[1] + box[3]), 'confidence': round(float(score), 3), 'emotion': emotion} for box, score, emotion in zip([i['box'] for i in results], [i['confidence'] for i in results], [i['emotion'] for i in results])], "image": f"/results/{ID}.jpg"}
    

@router.post("/infer-video", summary='Detect emotions of faces in video and return marked video', response_description="Something here")
async def emotion_detection_infer_video(file: UploadFile = File(...)):
    video_path = get_result_video(file, emotion_model, idx_to_class, test_transforms, device, detector)
    return {"video": video_path}