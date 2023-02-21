import uuid

import cv2 as cv
from fastapi import APIRouter, File

from ..files.Human_detection import (DetectorAPI, draw_bounding_box_on_image,
                                     get_image_with_cv2)

human_model = DetectorAPI()

router = APIRouter(
    prefix="/human",
    tags=["human"],
)


@router.post("/infer-image")
async def human_detection_infer_image(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    boxes, scores, classes, _ = human_model.processFrame(input_image)
    # draw boxes on image
    result = draw_bounding_box_on_image(input_image, boxes, scores, classes)
    # generate random UUID
    ID = uuid.uuid4()
    # return [ID]
    cv.imwrite(f"./ui/results/{ID}.jpg", result)
    return {"result": f"ui/results/{ID}.jpg"}

@router.post("/infer-json")
async def human_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    boxes, scores, _, num = human_model.processFrame(input_image)
    return {"result": [{'x_min': int(box[0]), 'y_min': int(box[1]), 'x_max': int(box[2]), 'y_max': int(box[3]), 'confidence': round(float(score), 3), 'name': 'person'} for box, score in zip(boxes[:num], scores[:num]) if score > 0.7]}
