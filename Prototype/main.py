from callbacks import Callbacks
from dataset_loader import get_transforms, LogoDetection
from fast_RCNN import Trainer
from imports import *
from yolo2coco import create_coco_file


if __name__ == "__main__":
    save_dir = (
        "dataset/"  # Target folder, used to save the generated coco format file
    )
    class_txt_path = (
        "names.txt"  # category file, one class per line, create by yourself
    )

    dataset_dirs = ["train", "val", "test"]
    for dir in dataset_dirs:
        label_dir = f"dataset/{dir}/labels" # The folder where the yolo format label is located
        img_dir = f"dataset/{dir}/images"  # The folder where the picture is located
        create_coco_file(save_dir + dir, class_txt_path, label_dir, img_dir)

    dataset_path = "dataset/"
    coco = COCO(os.path.join(dataset_path, "train\images", "_annotations.coco.json"))
    categories = coco.cats
    n_classes = len(categories.keys())
    classes = [i[1]["name"] for i in categories.items()]
    train_dataset = LogoDetection(root=dataset_path, transforms=get_transforms(True))
    val_dataset = LogoDetection(
        root=dataset_path, split="val\images", transforms=get_transforms(False)
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(device))
    model = Trainer(device)
    metric = MeanAveragePrecision(iou_type="bbox").to(device)
    train_acc = []
    val_acc = []
    min_lr = 1e-2

    num_epochs = 5
    callbacks = Callbacks(model.optimizer, val_acc, 0.005)


    for epoch in range(1, num_epochs + 1):
        train_acc.append(model.train_one_epoch(train_loader, device, epoch))
        val_acc.append(model.val_one_epoch(val_loader, device, metric, epoch))

        callbacks.save_weights(model.model, epoch, ".")

        if len(val_acc) > 2:
            if callbacks.EarlyStopping(min_lr):
                callbacks.save_weights(model.model, epoch, ".")
                break

        if epoch % 5 == 0:
            callbacks.save_weights(model.model, epoch, ".")
            callbacks.dropLR(10)
