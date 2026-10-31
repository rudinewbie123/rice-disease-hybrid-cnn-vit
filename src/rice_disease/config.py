from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config
