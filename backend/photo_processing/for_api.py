import cv2
import torch
from torchvision import datasets, models
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt


classes = ["tinkoff", "naumen", "ussc", "rostelecom", "sber", "gosuslugi"]
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
    torch.load(
        "photo_processing\model_weights_0910.pth", map_location=torch.device("cpu")
    )
)
device = torch.device("cpu")
model.to(device)
model.eval()


def single_image_prediction(image_path: str, save_path: str, threshold: float) -> None:
    image = cv2.imread(image_path)
    print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                classes[i - 1]
                for i in pred["labels"][pred["scores"] > threshold].tolist()
            ],
            width=4,
            colors="green",
        ).permute(1, 2, 0)
    )
    print(pred)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
