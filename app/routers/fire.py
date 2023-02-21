import io
import json
from fastapi import File, APIRouter
import base64

from PIL import Image
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from ..files.segmentation import get_image_from_bytes, get_yolov5



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
    prefix="/fire",
    tags=["fire"],
)


# @api_app.post("/save_file")
# async def upload_file(myfile: bytes = File(...)):
#     return {"size" : len(myfile)}
#     file_location = f"./a.dat"
#     with open(file_location, "wb") as file_object:
#         file_object.write(myfile)
#     return {"info": f"file saved"}

@router.post("/object-to-json")
async def detect_fire_return_json_result(file: bytes = File(...)):
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
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
        img_base64 = base64.b64encode(bytes_io.getvalue())
    return {"img_base64": img_base64}
    # return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
