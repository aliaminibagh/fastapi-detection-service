import json
import uuid

from fastapi import APIRouter, File, UploadFile

from ..files.utils_local import get_image_from_bytes, get_yolov5, get_video_from_bytes
from ..files.yolo_video import OD

model = get_yolov5(name='knife')


router = APIRouter(
    prefix="/knife",
    tags=["knife"],
)





@router.post("/infer-image", summary='Detect knives in image and return json', response_description="Something here")
async def knife_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    results.render()
    ID = uuid.uuid4()
    results.save(save_dir=f"./ui/results/{ID}")
    return {"result": detect_res, "image": f"/results/{ID}/image0.jpg"}


@router.post("/infer-video", summary='Detect knives in video and return json', response_description="Something here")
async def knife_detection_infer_json(file: UploadFile = File(...)):
    input_video , filename = get_video_from_bytes(file)
    detector = OD (capture_index = filename, model = model)
    video_path = detector()
    return {"video": video_path}