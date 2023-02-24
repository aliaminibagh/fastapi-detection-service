import json
import uuid

from fastapi import APIRouter, File

from ..files.utils import get_image_from_bytes, get_yolov5

model = get_yolov5(name="arms")


router = APIRouter(
    prefix="/arms",
    tags=["arms"],
)



@router.post("/infer-image", summary='Detect pistol in image and return json', response_description="Something here")
async def pistol_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    results.render()
    ID = uuid.uuid4()
    results.save(save_dir=f"./ui/results/{ID}")
    return {"result": detect_res, "image": f"/results/{ID}/image0.jpg"}
