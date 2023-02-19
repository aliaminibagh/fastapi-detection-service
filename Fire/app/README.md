# yolov5-fastapi
Machine Learning Model API using YOLOv5 with FAST API

### Getting start for this project

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```


#http://0.0.0.0:8000/docs#/ i



#Docker

docker build -t yolov5-fastapi:0.0.1 .

docker run -p 8080:8000 yolov5-fastapi:0.0.1
