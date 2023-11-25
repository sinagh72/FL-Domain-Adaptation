import os
from time import sleep
import flwr as fl
import torch
from dotenv import load_dotenv

from FedMim.client import client_fn_Mim
from FedMim.simmim import SimMimWrapper
from fl_strategy import FedAvgStrategy
from utils.utils import weighted_average, set_seed


def main(net, server_port) -> None:
    # Define strategy
    strategy = FedAvgStrategy(
        net=net,
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
        on_evaluate_config_fn=eval_config,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=3,  # Never sample less than num_clients for training
        min_evaluate_clients=3,  # Never sample less than num_clients for evaluation
        # # Minimum number of clients that need to be connected to the server before a training round can start
        min_available_clients=3,
        # fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address=f"0.0.0.0:{server_port}",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "epochs": 1,
        "patience": 10,
        "monitor": "val_loss",
        "mode": "min",
        "clients": 3,
        "batch_size": 32,
    }
    return config


def eval_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "clients": 3,
        "epochs": 1,
        "patience": 5,
        "monitor": "val_auc",
        "mode": "max",
    }
    return config


def simulation_main(net, client_fn) -> None:
    # Create FedAvg strategy
    strategy = FedAvgStrategy(
        net=net,
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
        on_evaluate_config_fn=eval_config,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 50% of available clients for evaluation
        # min_fit_clients=1,  # Never sample less than num_clients for training
        # min_evaluate_clients=1,  # Never sample less than num_clients for evaluation
        # # Minimum number of clients that need to be connected to the server before a training round can start
        # min_available_clients=1,
        # fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        ray_init_args={"num_gpus": 7, "num_cpus": os.cpu_count() - 4},
        client_resources={"num_gpus": 7 // NUM_CLIENTS, "num_cpus": (os.cpu_count() - 4)//NUM_CLIENTS},
    )


if __name__ == "__main__":
    trial = 10
    set_seed(10)
    NUM_CLIENTS = 3
    load_dotenv(dotenv_path="../data/.env")
    server_port = os.getenv('SERVER_PORT')

    for i in range(0, trial):
        simim = SimMimWrapper(lr=5e-4,
                              warmup_lr=5e-7,
                              wd=0.05,
                              min_lr=5e-6,
                              epochs=100,
                              warmup_epochs=10,
                              )
        # for i in range(1, 11):
        simulation_main(net=simim, client_fn=client_fn_Mim)
        # main(model, server_port)
        torch.cuda.empty_cache()
        sleep(1)
