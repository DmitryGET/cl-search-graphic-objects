from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

# Загружаем модель YOLO
yolo_model = YOLO("best.pt")

def detect_and_blur_logos(image_path, logo_class):
    """ Использует YOLO для поиска логотипов заданного класса и их размытия. """
    # Получаем имена классов из модели
    class_names = yolo_model.names
    target_class_id = [k for k, v in class_names.items() if v == logo_class][0]

    # Выполняем предсказание только для указанного класса
    results = yolo_model(image_path, classes=[target_class_id])
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # Координаты bounding box

    # Загружаем изображение для обработки
    main_image = cv2.imread(image_path)

    # Размываем найденные логотипы
    for (x1, y1, x2, y2) in boxes:
        roi = main_image[y1:y2, x1:x2]
        small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        main_image[y1:y2, x1:x2] = pixelated
        cv2.rectangle(main_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    output_path = os.path.join("static", "output.png")
    cv2.imwrite(output_path, main_image)
    return output_path

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/new_predict", response_class=HTMLResponse)
async def new_predict():
    return RedirectResponse(url="/", status_code=303)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), logo_class: str = Form(...)):
    file_location = os.path.join("uploads", file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Размываем логотипы выбранного класса
    output_image_path = detect_and_blur_logos(file_location, logo_class)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": f"/static/output.png"
    })