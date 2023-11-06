from imports import *


class Callbacks:
    def __init__(self, optimizer, val_acc, diff):
        self.optimizer = optimizer
        self.val_acc = val_acc
        self.diff = diff

    def dropLR(self, coef):
        if (self.val_acc[-1]["map_75"] - self.val_acc[-2]["map_75"]) < self.diff:
            self.optimizer.param_groups[0]["lr"] *= coef

    def EarlyStopping(self, threshold):
        if (
            self.val_acc[-1]["map_75"] - self.val_acc[-2]["map_75"]
        ) < self.diff and self.optimizer.param_groups[0]["lr"] == threshold:
            return True
        return

    def save_weights(self, model, epoch, save_path):
        torch.save(model.state_dict(), f"{save_path}/model_{epoch}_epochs.pth")
