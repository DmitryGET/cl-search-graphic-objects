from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO
import uuid

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

# Load YOLO model
yolo_model = YOLO("best.pt")


def detect_and_blur_logos(image_path, logo_class):
    """Detects and blurs logos of the specified class, returns both boxed and clean versions."""
    # Get class names and target class ID
    class_names = yolo_model.names
    target_class_id = [k for k, v in class_names.items() if v == logo_class][0]

    # Perform prediction for the specified class
    results = yolo_model(image_path, classes=[target_class_id])
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Load image for processing
    main_image = cv2.imread(image_path)
    clean_image = main_image.copy()  # For download without boxes

    # Blur detected logos
    for (x1, y1, x2, y2) in boxes:
        # Blur for both images
        for img in [main_image, clean_image]:
            roi = img[y1:y2, x1:x2]
            small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            img[y1:y2, x1:x2] = pixelated
        # Draw rectangle only on main_image (for display)
        cv2.rectangle(main_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    output_path_boxed = os.path.join("static", f"output_{unique_id}_boxed.png")
    output_path_clean = os.path.join("static", f"output_{unique_id}_clean.png")

    # Save both versions
    cv2.imwrite(output_path_boxed, main_image)
    cv2.imwrite(output_path_clean, clean_image)
    return output_path_boxed, output_path_clean


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/new_predict", response_class=HTMLResponse)
async def new_predict():
    return RedirectResponse(url="/", status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, files: list[UploadFile] = File(...), logo_class: str = Form(...)):
    processed_images = []

    # Process each uploaded file
    for file in files:
        file_location = os.path.join("uploads", file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image and get both versions
        output_path_boxed, output_path_clean = detect_and_blur_logos(file_location, logo_class)
        processed_images.append({
            "boxed_path": f"/{output_path_boxed}",
            "clean_path": f"/{output_path_clean}",
            "filename": file.filename
        })

    return templates.TemplateResponse("result.html", {
        "request": request,
        "images": processed_images
    })


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("static", filename)
    print(f"Запрашиваемый файл: {file_path}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл {filename} не найден")
    # Извлекаем оригинальное имя файла или используем префикс
    download_filename = f"blurred_{filename}"  # Или передать оригинальное имя через параметр
    return FileResponse(file_path, filename=download_filename, media_type='application/octet-stream')