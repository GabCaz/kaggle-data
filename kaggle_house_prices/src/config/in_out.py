import yaml
from pathlib import Path

_PATH_TO_CONFIG = Path(__file__).resolve().parent


def read_yaml(fname: str) -> dict:
    with open(_PATH_TO_CONFIG / fname, "r") as file:
        config = yaml.safe_load(file)
    return config


def write_yaml(config: dict, fname: str) -> None:
    with open(_PATH_TO_CONFIG / fname, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
