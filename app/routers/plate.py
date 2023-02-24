
from fastapi import APIRouter, File

from ..files.utils import get_image_with_cv2

router = APIRouter(
    prefix="/plate",
    tags=["plate"],
)

@router.post("/infer-image")
async def plate_detection_infer_image(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    return {"result": "ok"}



