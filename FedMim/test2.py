import os

import torch
import torchvision
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping
from torchvision.models import Swin_V2_S_Weights

from FedMim.client import count_classes_and_compute_weights
from FedMim.fedmim import FedMim
from FedMim.simmim import SimMimWrapper
from fl_config import get_dataloaders
from models.resnet import ResNet
from utils.data_handler import get_datasets_classes, get_datasets_full_classes
from utils.transformations import get_finetune_transformation, SimMIMTransform
from utils.utils import set_seed, get_hyperparameters
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def imshow(img_tensor):
    img = img_tensor.clone().detach()  # Clone and detach from the current graph
    img = transforms.ToPILImage()(img)  # Convert to PIL image
    return img


if __name__ == "__main__":
    max_epochs = 200
    set_seed(10)
    encoder = torchvision.models.swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT, progress=True)
    load_dotenv(dotenv_path="../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    architecture = "simim"
    batch_size = 1
    img_size = 256
    client_name = str(os.getenv('CLIENT_NAME'))
    cls_kermany_classes, cls_srinivasan_classes, cls_oct500_classes = get_datasets_classes()
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_full_classes()
    simmim_transform = SimMIMTransform(img_size, model_patch_size=4,
                                       mask_patch_size=32,
                                       mask_ratio=0.7,
                                       mean=0.5,
                                       std=0.5
                                       )
    train_loader, val_loader, _, _ = get_dataloaders(cid="1",
                                                     dataset_path=DATASET_PATH,
                                                     batch_size=batch_size,
                                                     kermany_classes=kermany_classes,
                                                     srinivasan_classes=srinivasan_classes,
                                                     oct500_classes=oct500_classes,
                                                     img_transforms=simmim_transform,
                                                     )
    for i in train_loader:
        x, mask = i["img"]
        mask = mask.squeeze()
        x = x.squeeze()
        for i in range(mask.size(0)):  # Loop through rows of the mask
            for j in range(mask.size(1)):  # Loop through columns of the mask
                if mask[i, j] == 0:
                    # Set corresponding 4x4 patch in the tensor to 0 (black)
                    x[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = 0
                # If mask[i, j] is 1, do nothing (keep the original values)

        plt.figure()  # Create a new figure
        img = imshow(x)  # Show image
        plt.imshow(img, cmap="gray")
        plt.title("Label")
        plt.show()

    simim = SimMimWrapper(lr=5e-4,
                          warmup_lr=5e-7,
                          wd=0.05,
                          min_lr=5e-6,
                          epochs=100,
                          warmup_epochs=10,
                          )
    early_stopping = EarlyStopping(monitor="val_auc", patience=5, verbose=False,
                                   mode="max")
    trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=10,
                         callbacks=[early_stopping],
                         enable_checkpointing=False,
                         # log_every_n_steps=config["log_n_steps"],
                         )
    trainer.fit(model=simim, train_dataloaders=train_loader, val_dataloaders=val_loader)
