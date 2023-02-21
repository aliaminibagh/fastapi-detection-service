from fastapi import File, APIRouter
import base64
from starlette.responses import Response
from PIL import Image
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import io
from ..files.Human_detection import DetectorAPI, get_image_with_cv2, draw_bounding_box_on_image


# from fastapi.middleware.cors import CORSMiddleware


human_model = DetectorAPI()

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
    prefix="/human",
    tags=["human"],
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
    input_image = get_image_with_cv2(file)
    boxes, scores, _, num = human_model.processFrame(input_image)
    return {"result": [{'x_min': int(box[0]), 'y_min': int(box[1]), 'x_max': int(box[2]), 'y_max': int(box[3]), 'confidence': round(float(score), 3), 'name': 'person'} for box, score in zip(boxes[:num], scores[:num]) if score > 0.7]}


@router.post("/object-to-base64")
async def detect_fire_return_img(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    boxes, scores, classes, _ = human_model.processFrame(input_image)
    results = draw_bounding_box_on_image(input_image, boxes, scores, classes)
    img_base64 = base64.b64encode(results).decode('utf-8')
    return {"img_base64": img_base64}
    # return Response(content=bytes_io.getvalue(), media_type="image/jpeg")


@router.post("/object-to-img")
async def detect_fire_return_img(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    boxes, scores, classes, num = human_model.processFrame(input_image)
    print(boxes)
    results = draw_bounding_box_on_image(input_image, boxes, scores, classes)
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(results)
    img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
