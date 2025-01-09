"""Flower-with-TensorFlow: A Flower / TensorFlow app."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flower_with_tensorflow.strategy import FedCustom


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=FedCustom(), config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
