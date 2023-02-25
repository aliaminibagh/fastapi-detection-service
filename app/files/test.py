from fastapi import FastAPI, File, UploadFile
import uuid
from utils_local import get_image_with_cv2, get_yolov5, get_video_from_bytes, get_video_from_bytes_temp, get_yolov8
from yolo_video import OD

model = get_yolov8(name= "smoke")


print(model)
app = FastAPI()



@app.post("/video")
async def video(file: UploadFile = File(...)):
    input_video , filename = get_video_from_bytes(file)
    detector = OD (capture_index = filename, model_name = "knife")
    video_path = detector()
    # results.render()
    return {"video": video_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=3001, reload=True)