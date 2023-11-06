from imports import *


class Trainer:
    def __init__(self, device):
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        in_features = (
            self.model.roi_heads.box_predictor.cls_score.in_features
        )  # we need to change the head
        self.model.roi_heads.box_predictor = (
            models.detection.faster_rcnn.FastRCNNPredictor(in_features, 7)
        )
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            self.params, lr=1e-4, momentum=0.9, weight_decay=1e-4
        )
        self.model = self.model.to(device)

    def train_one_epoch(self, train_loader, device, epoch):
        self.model.to(device)
        self.model.train()
        all_losses = []
        all_losses_dict = []

        for images, targets in tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [
                {k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = self.model(
                images, targets
            )  # the model computes the loss automatically if we pass in targets
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
            loss_value = losses.item()

            all_losses.append(loss_value)
            all_losses_dict.append(loss_dict_append)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping trainig"
                )  # train if loss becomes infinity
                print(loss_dict)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
        print(
            "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                epoch,
                self.optimizer.param_groups[0]["lr"],
                np.mean(all_losses),
                all_losses_dict["loss_classifier"].mean(),
                all_losses_dict["loss_box_reg"].mean(),
                all_losses_dict["loss_rpn_box_reg"].mean(),
                all_losses_dict["loss_objectness"].mean(),
            )
        )
        return all_losses_dict

    def val_one_epoch(self, val_loader, device, metric, epoch):
        self.model.eval()

        for images, targets in tqdm(val_loader):
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: torch.tensor(v).to(device)
                    for k, v in t.items()
                    if k in ["boxes", "labels"]
                }
                for t in targets
            ]

            with torch.no_grad():
                prediction = self.model(images)

            metric.update(prediction, targets)
            result = metric.compute()

        metric.reset()
        result_dict = pd.DataFrame(result)  # for printing
        print(
            "Validation: Epoch {}, lr: {:.6f}, mAP: {:.4f}, mAP50: {:.4f}, mAP75: {:.4f}".format(
                epoch,
                self.optimizer.param_groups[0]["lr"],
                result["map"],
                result["map_50"],
                result["map_75"],
            )
        )
        return result
