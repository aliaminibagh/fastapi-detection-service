import uuid

import cv2
from fastapi import APIRouter, File

from ..files.Human_detection import get_image_with_cv2
from ..files.segmentation import get_image_from_bytes, get_yolov8

idx_to_class = {0.0:"fire", 1.0: "default", 2.0: "smoke"}


model = get_yolov8()



router = APIRouter(
    prefix="/smoke",
    tags=["smoke"],
)


@router.post("/infer-image")
async def smoke_detection_infer_image(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    results = model(input_image)
    # results.render()
    ID = uuid.uuid4()
    boxes = results[0].boxes.data.tolist()
    for box in boxes:
        cv2.rectangle(input_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (36,255,12), 2)
        cv2.putText(input_image, idx_to_class[int(box[5])], (int(box[0] + 10), int(box[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.imwrite(f"./ui/results/{ID}.jpg", input_image)
    return {"result": f"ui/results/{ID}.jpg"}

@router.post("/infer-json")
async def smoke_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model.predict(source=input_image)
    boxes = results[0].boxes.data.tolist()
    return {"result": [{"x_min" : box[0], "y_min" : box[1],"x_max" : box[2],"y_max" : box[3], "conf": round(box[4], 3), "class":idx_to_class[box[5]]} for box in boxes]}

