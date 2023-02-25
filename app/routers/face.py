
from fastapi import APIRouter, File

from ..files.utils_local import get_image_with_cv2

router = APIRouter(
    prefix="/face",
    tags=["face"],
)

@router.post("/infer-image")
async def face_detection_infer_image(file: bytes = File(...)):
    input_image = get_image_with_cv2(file)
    return {"result": "همه چی داره خوب کار میکنه ! من فقط منتظر یک جواب از طرف سرور برای نمایش هستم."}
