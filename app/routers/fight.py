import io
import json

from fastapi import FastAPI, File, APIRouter


from PIL import Image
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from ..files.segmentation import get_image_from_bytes, get_yolov5


# from fastapi.middleware.cors import CORSMiddleware


model = get_yolov5()

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
    prefix="/fight",
    tags=["fight"],
)

@router.get("/test")
async def test():
    return {"result": "ok"}


