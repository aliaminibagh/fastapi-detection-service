import io
import json

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from .segmentation import get_image_from_bytes, get_yolov5

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
api_app  = FastAPI(
    # middleware=middleware,
    title="Fire Detection API Service",
    description="""Fire Detection Service""",
    version="0.0.1",
    contact={
        "name": "Ali Amini Bagh",
        "email": "aliaminibagh@gmail.com",
    }
)

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app = FastAPI(title="main app")
app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="./ui", html=True), name="ui")


@api_app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=f"/api/docs/", status_code=303)

@api_app.get("/echo/{name}")
async def echome(name:str):
    return {"result": name}

@api_app.post("/save_file")
async def upload_file(myfile: bytes = File(...)):
    return {"size" : len(myfile)}
    file_location = f"./a.dat"
    with open(file_location, "wb") as file_object:
        file_object.write(myfile)
    return {"info": f"file saved"}

@api_app.post("/object-to-json")
async def detect_fire_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}

import base64

@api_app.post("/object-to-img")
async def detect_fire_return_img(file: bytes =  File(...)):
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

