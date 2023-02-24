import uuid

import cv2 as cv
from fastapi import APIRouter, File

from ..files.human_utils import DetectorAPI
from ..files.utils import draw_bounding_box_on_image, get_image_with_cv2

human_model = DetectorAPI()

router = APIRouter(
    prefix="/human",
    tags=["human"],
)




@router.post("/infer-image", summary='Detect humans in image and return json', response_description="Something here")
async def human_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    boxes, scores, classes, num = human_model.processFrame(input_image)
    result = draw_bounding_box_on_image(input_image, boxes, scores, classes)
    ID = uuid.uuid4()
    cv.imwrite(f"./ui/results/{ID}.jpg", result)
    return {"result": [{'x_min': int(box[0]), 'y_min': int(box[1]), 'x_max': int(box[2]), 'y_max': int(box[3]), 'confidence': round(float(score), 3), 'name': 'person'} for box, score in zip(boxes[:num], scores[:num]) if score > 0.7], "image": f"/results/{ID}.jpg"}
