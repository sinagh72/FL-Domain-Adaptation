import os

from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping

from FedMim.fedmim import FedMim
from FedMim.simmim import SimMimWrapper
from fl_config import get_dataloaders
from utils.data_handler import get_datasets_classes, get_datasets_full_classes
from utils.transformations import get_finetune_transformation
from utils.utils import set_seed, get_hyperparameters
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch as pl


if __name__ == "__main__":
    max_epochs = 200
    set_seed(10)

    load_dotenv(dotenv_path="../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    architecture = "simim"
    batch_size = 32
    img_size = 128
    client_name = str(os.getenv('CLIENT_NAME'))
    cls_kermany_classes, cls_srinivasan_classes, cls_oct500_classes = get_datasets_classes()
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_full_classes()
    cls_train_loader, cls_val_loader, test_loader, classes = get_dataloaders(cid="0",
                                                                             dataset_path=DATASET_PATH,
                                                                             batch_size=batch_size,
                                                                             kermany_classes=cls_kermany_classes,
                                                                             srinivasan_classes=cls_srinivasan_classes,
                                                                             oct500_classes=cls_oct500_classes,
                                                                             img_transforms=get_finetune_transformation(
                                                                                 img_size),
                                                                             )
    # preparing config
    step_size = len(cls_train_loader)//batch_size * 5
    lr = 1e-4
    wd = 1e-6

    simim = SimMimWrapper(lr=5e-4,
                          warmup_lr=5e-7,
                          wd=0.05,
                          min_lr=5e-6,
                          epochs=300,
                          warmup_epochs=20,
                          )
    param = get_hyperparameters(client_name, "ViT")

    model = FedMim(encoder=simim.model.encoder,
                   wd=param["wd"],
                   lr=param["lr"],
                   beta1=param["beta1"],
                   beta2=param["beta2"],
                   step_size=len(cls_train_loader) * batch_size // 2, gamma=0.5,
                   classes=classes
                   )

    early_stopping = EarlyStopping(monitor="val_auc", patience=5, verbose=False,
                                   mode="max")
    trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=10,
                         callbacks=[early_stopping],
                         enable_checkpointing=False,
                         # log_every_n_steps=config["log_n_steps"],
                         )
    trainer.fit(model=model, train_dataloaders=cls_train_loader, val_dataloaders=cls_val_loader)


