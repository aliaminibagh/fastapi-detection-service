import json
import uuid

from fastapi import APIRouter, File, UploadFile

from ..files.utils_local import get_image_from_bytes, get_yolov5, get_video_from_bytes
from ..files.yolo_video import OD

model = get_yolov5(name="fire")


router = APIRouter(
    prefix="/fire",
    tags=["fire"],
)





@router.post("/infer-image", summary='Detect fire in image and return json', response_description="Something here")
async def fire_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(
        orient="records")
    detect_res = json.loads(detect_res)
    results.render()
    ID = uuid.uuid4()
    results.save(save_dir=f"./ui/results/{ID}")
    if len(detect_res) == 0:
        return {"result": "در تصویر آتش یافت نشد.", "image": f"/results/{ID}/image0.jpg"}
    return {"result": detect_res, "image": f"/results/{ID}/image0.jpg"}


@router.post("/infer-video", summary='Detect fire in video and return json', response_description="Something here")
async def fire_detection_infer_video(file: UploadFile = File(...)):
    input_video, filename = get_video_from_bytes(file)
    detector = OD(capture_index=filename, model_name="fire")
    video_path = detector()
    return {"video": video_path}