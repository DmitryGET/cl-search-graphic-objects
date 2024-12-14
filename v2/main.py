from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from ultralytics import YOLO
from PIL import Image
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="/app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="/app/uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")


os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)


model = YOLO("best.pt")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), confidence: float = Form(...)):
    file_location = os.path.join("uploads", file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Обработка изображения моделью YOLOv8
    results = model.predict(source=file_location, conf=confidence)
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image_path = os.path.join("static", f"annotated_{file.filename}")
    Image.fromarray(annotated_image).save(annotated_image_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": f"/static/annotated_{file.filename}"
    })

@app.get("/new_predict", response_class=HTMLResponse)
async def new_predict():
    return RedirectResponse(url="/", status_code=303)
