import base64
import io
import json
import uuid

from fastapi import APIRouter, File, Response
from PIL import Image
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from ..files.segmentation import get_image_from_bytes, get_yolov5

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




@router.post("/infer-image")
async def fire_detection_infer_image(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()
    ID = uuid.uuid4()
    results.save(save_dir = f"./ui/results/{ID}")
    return {"result": f"ui/results/{ID}/image0.jpg"}

@router.post("/infer-json")
async def fire_detection_infer_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(
        orient="records")  
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@router.post("/object-to-base64")
async def detect_fire_return_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
        img_base64 = base64.b64encode(bytes_io.getvalue())
    return {"img_base64": img_base64}
    # return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

@router.post("/object-to-img")
async def detect_fire_return_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
        img_base64 = base64.b64encode(bytes_io.getvalue())
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
