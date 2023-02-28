
from fastapi import APIRouter, File, UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from ..files.utils_local import predict_on_live_video

tf.compat.v1.disable_eager_execution()
model = load_model("./app/files/model/fight.h5")

router = APIRouter(
    prefix="/fight",
    tags=["fight"],
)

@router.post("/infer-video", summary='Detect fight in video and return json', response_description="Something here")
async def fight_detection_infer_video(file: UploadFile = File(...)):
    video_path = predict_on_live_video(binary_video=file, model=model)
    return {"video": video_path}



