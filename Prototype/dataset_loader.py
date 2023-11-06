from imports import *


def get_transforms(train=False):
    if train:
        transform = A.Compose(
            [
                A.Resize(600, 600),  # our input size can be 600px
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.1),
                A.ColorJitter(p=0.1),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco"),
        )
    else:
        transform = A.Compose(
            [A.Resize(600, 600), ToTensorV2()],  # our input size can be 600px
            bbox_params=A.BboxParams(format="coco"),
        )
    return transform


class LogoDetection(datasets.VisionDataset):
    def __init__(
        self,
        root,
        split="train/images",
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split  # train, valid, test
        self.coco = COCO(
            os.path.join(root, split, "_annotations.coco.json")
        )  # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [
            t["bbox"] + [t["category_id"]] for t in target
        ]  # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed["image"]
        boxes = transformed["bboxes"]

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}  # here is our transformed target
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor(
            [t["category_id"] for t in target], dtype=torch.int64
        )
        targ["image_id"] = torch.tensor([t["image_id"] for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (
            boxes[:, 2] - boxes[:, 0]
        )  # we have a different area
        targ["iscrowd"] = torch.tensor(
            [t["iscrowd"] for t in target], dtype=torch.int64
        )
        return image.div(255), targ  # scale images

    def __len__(self):
        return len(self.ids)
