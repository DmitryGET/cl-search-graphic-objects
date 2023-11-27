import os
import shutil

import cv2
import torch
from django.conf import settings
from torchvision import datasets, models
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
import matplotlib.pyplot as plt


classes_logo = ["tinkoff", "naumen", "ussc", "rostelecom", "sber", "gosuslugi"]
classes_banking = ["mastercard", "mir", "unionpay", "visa"]
model = (
    models.detection.fasterrcnn_mobilenet_v3_large_fpn()
)  # we do not specify ``weights``, i.e. create untrained model
in_features = (
    model.roi_heads.box_predictor.cls_score.in_features
)  # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, 7
)
model.load_state_dict(
    torch.load(r"photo_processing\banking.pth", map_location=torch.device("cpu"))
)
device = torch.device("cpu")
model.to(device)
model.eval()


def single_image_prediction(
    image_path: str, save_path: str, threshold: float, model_type
):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_type == "YOLO":
        weights_path = r"photo_processing\yolo_banking.pt"
        model1 = YOLO(weights_path)
        project = settings.MEDIA_ROOT + r"\processed_photos"
        name = "pic"
        tt = settings.MEDIA_ROOT + r"\processed_photos\pic"
        try:
            # Перед удалением удостоверьтесь, что путь существует и является директорией
            if os.path.exists(tt) and os.path.isdir(tt):
                shutil.rmtree(tt)
        except Exception as e:
            print(f"Ошибка при удалении директории {tt}: {e}")
        a = model1.predict(
            image_path, iou=0.95, conf=0.5, project=project, name=name, save=True
        )
        return tt
    else:
        transform = T.Compose([T.ToTensor()])
        tensor = transform(image)

        img_int = torch.tensor(tensor * 255, dtype=torch.uint8)
        with torch.no_grad():
            prediction = model([tensor.to(device)])
            pred = prediction[0]

        fig = plt.figure(figsize=(14, 10))
        plt.axis("off")
        plt.imshow(
            draw_bounding_boxes(
                img_int,
                pred["boxes"][pred["scores"] > threshold],
                [
                    classes_banking[i - 1]
                    for i in pred["labels"][pred["scores"] > threshold].tolist()
                ],
                width=4,
                colors="green",
            ).permute(1, 2, 0)
        )
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    return "Fast"
