from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/")
def home():
    return {"message": "AI 서버 실행중"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(filepath)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 2)
            })

    return {
        "success": True,
        "detections": detections
    }
