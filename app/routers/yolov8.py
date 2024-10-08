import uuid

import cv2
from fastapi import APIRouter, File, UploadFile

from ..files.utils_local import get_image_with_cv2, get_yolov8, get_video_from_bytes
from ..files.yolo_video import OD

model = get_yolov8()


router = APIRouter(
    prefix="/yolov8",
    tags=["yolov8"],
)




@router.post("/infer-image", summary='Detect objects in image and return json', response_description="Something here")
async def yolov8_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    results = model.predict(source=input_image)
    ID = uuid.uuid4()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cv2.rectangle(input_image, (int(box.data[0][0]) , int(box.data[0][1])), (int(box.data[0][2]), int(box.data[0][3])), (36, 255, 12), 2)
            cv2.putText(input_image, model.names[int(box.data[0][5])], (int(box.data[0][0] + 10), int(box.data[0][1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imwrite(f"./ui/results/{ID}.jpg", input_image)
    return {"result": [{"x_min": int(box.data[0][0]), "y_min": int(box.data[0][1]), "x_max": int(box.data[0][2]), "y_max": int(box.data[0][3]), "conf": round(float(box.data[0][4]), 3), "class": model.names[int(box.data[0][5])]} for box in boxes], "image": f"/results/{ID}.jpg"}


@router.post("/infer-video", summary='Detect objects in video and return json', response_description="Something here")
async def yolov8_detection_infer_video(file: UploadFile = File(...)):
    input_video, filename = get_video_from_bytes(file)
    detector = OD(capture_index=filename, model_name= "yolov8x", yolo_version=8)
    video_path = detector()
    return {"video": video_path}