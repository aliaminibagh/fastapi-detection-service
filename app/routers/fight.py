
from fastapi import APIRouter

# from ..files.segmentation import get_yolov5

# model = get_yolov5()

router = APIRouter(
    prefix="/fight",
    tags=["fight"],
)

@router.get("/test")
async def test():
    return {"result": "ok"}


