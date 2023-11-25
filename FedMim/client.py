import copy
from time import sleep
import torch
from dotenv import load_dotenv
import flwr as fl
from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl
from FedMim.fedmim import FedMim
from FedMim.simmim import SimMimWrapper
from FedMim.simmim_transformation import SimMIMTransform
from fl_client import FlowerClient
import os
from fl_config import get_dataloaders, log_results
from utils.data_handler import get_datasets_classes, get_datasets_full_classes
from utils.transformations import get_finetune_transformation
from utils.utils import get_img_transformation, set_seed, get_hyperparameters


class FlowerClientMim(FlowerClient):
    def __init__(self, net, cls, cls_train_loader, cls_val_loader, test_loader,
                 masked_train_loader, masked_val_loader, client_name, architecture):
        super().__init__(net, masked_train_loader, masked_val_loader, test_loader, client_name, architecture)
        self.cls = cls
        self.cls_train_loader = cls_train_loader
        self.cls_val_loader = cls_val_loader

    def fit(self, parameters, config):
        """
        Receive model parameters from the server, train the model parameters on the local data,
        and return the (updated) model parameters to the server
        :param parameters:
        :param config: dictionary contains the fit configuration
        :return: local model's parameters, length train data,
        """
        set_seed()
        self.set_parameters(parameters, config)
        early_stopping = EarlyStopping(monitor=config["monitor"], patience=config["patience"], verbose=False,
                                       mode=config["mode"])
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=config["simim_epochs"],
                             callbacks=[early_stopping],
                             logger=False,
                             enable_checkpointing=False,
                             # log_every_n_steps=config["log_n_steps"],
                             )
        trainer.fit(model=self.net, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        return self.get_parameters(self.net.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        """
              Receive model parameters from the server, evaluate the model parameters on the local data,
              and return the evaluation result to the server
              :param parameters:
              :param config:
              :return:
              """
        self.set_parameters(parameters, config)
        early_stopping = EarlyStopping(monitor=config["monitor"], patience=config["patience"], verbose=False,
                                       mode=config["mode"])
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=config["cls_epochs"],
                             callbacks=[early_stopping],
                             logger=False,
                             enable_checkpointing=False,
                             )
        trainer.fit(model=self.cls, train_dataloaders=self.cls_train_loader, val_dataloaders=self.cls_val_loader)
        test_results = trainer.test(self.cls, self.test_loader, verbose=True)
        print("============================")
        log_results(classes=self.cls.hparams.classes,
                    results=test_results,
                    client_name=self.client_name,
                    architecture=self.architecture,
                    config=config)


def client_fn_Mim(cid: str) -> FlowerClientMim:
    """Creates a FlowerClient instance on demand
    Create a Flower client representing a single organization
    """
    set_seed(10)
    load_dotenv(dotenv_path="../data/.env")
    DATASET_PATH = os.getenv('DATASET_PATH')
    architecture = "simim"
    batch_size = 32
    img_size = 128
    client_name = str(os.getenv('CLIENT_NAME'))
    cls_kermany_classes, cls_srinivasan_classes, cls_oct500_classes = get_datasets_classes()
    kermany_classes, srinivasan_classes, oct500_classes = get_datasets_full_classes()
    cls_train_loader, cls_val_loader, test_loader, classes = get_dataloaders(cid=cid,
                                                                             dataset_path=DATASET_PATH,
                                                                             batch_size=batch_size,
                                                                             kermany_classes=cls_kermany_classes,
                                                                             srinivasan_classes=cls_srinivasan_classes,
                                                                             oct500_classes=cls_oct500_classes,
                                                                             img_transforms=get_finetune_transformation(
                                                                                 img_size),
                                                                             )
    simmim_transform = SimMIMTransform(img_size, model_patch_size=4,
                                       mask_patch_size=32,
                                       mask_ratio=0.3,
                                       mean=0.5,
                                       std=0.5
                                       )
    train_loader, val_loader, _, _ = get_dataloaders(cid=cid,
                                                     dataset_path=DATASET_PATH,
                                                     batch_size=batch_size,
                                                     kermany_classes=kermany_classes,
                                                     srinivasan_classes=srinivasan_classes,
                                                     oct500_classes=oct500_classes,
                                                     img_transforms=simmim_transform,
                                                     )

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
                   step_size=len(cls_train_loader) * batch_size * 5, gamma=0.5,
                   classes=classes
                   )
    return FlowerClientMim(simim, model, cls_train_loader, cls_val_loader, test_loader, train_loader, val_loader,
                           client_name=client_name, architecture=architecture)


if __name__ == "__main__":
    set_seed(10)
    server_ip = os.getenv('SERVER_IP')
    trials = 10
    param = get_hyperparameters(client_name, "ViT")
    # Model and data
    for i in range(0, trials):
        simim = SimMimWrapper(num_classes=len(classes),
                              lr=5e-4,
                              warmup_lr=5e-7,
                              wd=0.05,
                              min_lr=5e-6,
                              epochs=300,
                              warmup_epochs=20,
                              )

        model = FedMim(encoder=simim.model.encoder,
                       wd=param["wd"],
                       lr=param["lr"],
                       beta1=param["beta1"],
                       beta2=param["beta2"],
                       step_size=len(cls_train_loader * batch_size) // 2, gamma=0.5
                       )

        client = FlowerClientMim(simim, model, train_loader, val_loader, test_loader,
                                 client_name=client_name,
                                 architecture=architecture)
        fl.client.start_numpy_client(server_address=f"{server_ip}:{os.getenv('SERVER_PORT')}",
                                     client=client)
        torch.cuda.empty_cache()
        sleep(10)
