
from fastapi import APIRouter, File
import torch
from torchvision import transforms
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from mtcnn_cv2 import MTCNN
from ..files.Human_detection import get_image_with_cv2, draw_bounding_box_on_image
from ..files.segmentation import infer_emotions

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




middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

router = APIRouter(
    prefix="/emotions",
    tags=["emotions"],
)


@router.post("/object-to-json")
async def detect_emotions_return_json_result(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    results = infer_emotions(input_image, emotion_model,idx_to_class, test_transforms,device, detector)
    if len(results) == 0:
        return {"message": "No faces detected"}
    return ({f'Face_{num+1}': i['box'], 'Confidence': round(i['confidence'], 4), 'Emotion': i['emotion']} for num, i in enumerate(results))
