from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from scipy.spatial.distance import cosine
from PIL import Image

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
# Монтируем каталоги для статики и загрузок
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

# Загружаем модель YOLO
yolo_model = YOLO("best.pt")

# Загрузка предобученной модели ResNet для извлечения признаков
resnet = models.resnet34(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Убираем последний слой
resnet.eval()

# Трансформация изображений для подачи в ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image):
    """ Извлекает вектор признаков изображения с помощью ResNet. """
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension
    with torch.no_grad():
        features = resnet(image).squeeze().numpy()  # Извлекаем 2048-мерный вектор
    return features


def detect_logos(image_path):
    """ Использует YOLO для нахождения логотипов. """
    results = yolo_model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # Координаты bounding box
    return boxes


def match_logo(main_image_path, logo_path):
    """ Ищет логотип среди найденных объектов и заблюривает его. """
    main_image = cv2.imread(main_image_path)
    logo_image = cv2.imread(logo_path)

    # Извлекаем признаки логотипа
    logo_features = extract_features(cv2.cvtColor(logo_image, cv2.COLOR_BGR2RGB))

    # Ищем логотипы на основном изображении
    boxes = detect_logos(main_image_path)

    for (x1, y1, x2, y2) in boxes:
        cropped_logo = main_image[y1:y2, x1:x2]  # Вырезаем логотип
        cropped_features = extract_features(cv2.cvtColor(cropped_logo, cv2.COLOR_BGR2RGB))

        # Вычисляем косинусное расстояние
        similarity = cosine(logo_features, cropped_features)
        if similarity > 0.3:  # Если найденный логотип похож
            roi = main_image[y1:y2, x1:x2]
            small = cv2.resize(roi, (25, 25), interpolation=cv2.INTER_LINEAR)
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
async def predict(request: Request, file: UploadFile = File(...), logo: UploadFile = File(...)):
    # Сохраняем логотип и изображение
    file_location = os.path.join("uploads", file.filename)
    logo_location = os.path.join("uploads", logo.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(logo_location, "wb") as buffer:
        shutil.copyfileobj(logo.file, buffer)

    # Применяем обработку и размывание логотипов на изображении
    output_image_path = match_logo(file_location, logo_location)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": f"/static/output.png"
})