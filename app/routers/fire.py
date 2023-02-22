import json
import uuid

from fastapi import APIRouter, File
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from ..files.utils import get_image_from_bytes, get_yolov5

model = get_yolov5(name="fire")

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
    prefix="/fire",
    tags=["fire"],
)


@router.post("/infer-image",  summary='Detect fire in image and return marked image', response_description="Something here")
async def fire_detection_infer_image(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()
    ID = uuid.uuid4()
    results.save(save_dir=f"./ui/results/{ID}")
    return {"result": f"/results/{ID}/image0.jpg"}


@router.post("/infer-json", summary='Detect fire in image and return json', response_description="Something here")
async def fire_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(
        orient="records")
    detect_res = json.loads(detect_res)
    return {"result": detect_res}
