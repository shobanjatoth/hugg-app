from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
model = YOLO(r"E:\helmet-detection-fastapi\artifact\model_trainer\helmet_detector\weights\better.pt")
model.fuse()

def read_imagefile(file) -> np.ndarray:
    image = Image.open(BytesIO(file)).convert("RGB")
    return np.array(image)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/mobile", response_class=HTMLResponse)
def mobile_ui(request: Request):
    return templates.TemplateResponse("mobile_realtime.html", {"request": request})

@app.post("/predict/image")
def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(file.file.read())
    results = model(image)[0]
    annotated = results.plot()
    _, encoded_img = cv2.imencode(".jpg", annotated)
    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

@app.post("/predict/video")
def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        temp_in.write(file.file.read())
        temp_video_path = temp_in.name

    model.predict(source=temp_video_path, save=True, save_txt=False, project="video_output", name="predict", exist_ok=True)
    output_name = os.path.basename(temp_video_path)
    pred_path = os.path.join("video_output", "predict", output_name)

    return FileResponse(pred_path, media_type="video/mp4", filename="prediction.mp4")

@app.post("/predict/frame")
def predict_frame(file: UploadFile = File(...)):
    image = read_imagefile(file.file.read())
    results = model(image)[0]
    annotated = results.plot()
    _, encoded_img = cv2.imencode(".jpg", annotated)
    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


