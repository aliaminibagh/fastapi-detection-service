import cv2
from fastapi import FastAPI, File, UploadFile
import uuid
from utils_local import get_image_with_cv2, get_yolov5, get_video_from_bytes, get_video_from_bytes_temp, get_yolov8
from yolo_video import OD

model = get_yolov5(name= "fire")


classes = model.names
def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]

results = model("app/files/3.jpg")
print(results)
# app = FastAPI()
frame = cv2.imread('app/files/3.jpg')
labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
n = len(labels)
x_shape, y_shape = frame.shape[1], frame.shape[0]
for i in range(n):
    row = cord[i]
    if row[4] >= 0.3:
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

# @app.post("/video")
# async def video(file: UploadFile = File(...)):
#     input_video , filename = get_video_from_bytes(file)
#     detector = OD (capture_index = filename, model_name = "knife")
#     video_path = detector()
#     # results.render()
#     return {"video": video_path}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("test:app", host="0.0.0.0", port=3001, reload=True)