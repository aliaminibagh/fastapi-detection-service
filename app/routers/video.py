import uuid
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
import aiofiles
import asyncio
import os
import cv2
from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/video", tags=["video"])

#define an API endpoint which takes a video file as input and save all the frames of the video in a folder
@router.post("/save-frames", summary='Save all frames of a video in a folder', response_description="Something here")
async def save_frames(file: UploadFile = File(...)):
    #create a temporary file to save the video
    video = await file.read()
    counter = 0
    while True:
        is_read, frame = cv2.VideoCapture(video).read()
        if not is_read:
            break
        ID = uuid.uuid4()
        cv2.imwrite(f"./ui/results/{ID}/{counter}.jpg", frame)
        counter += 1
    return {"result": f"ui/results/{ID}"}

