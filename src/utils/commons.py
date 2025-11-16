import os
import yaml
import json
import torch
from src import logger
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError



def read_yaml(file_path: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox.
    Args:
        file_path (Path): Path to the YAML file.
    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox.
    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error reading the YAML file.
        BoxValueError: If there is an error converting the YAML to ConfigBox.
        Exception: For any other unexpected errors.
    """
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Error reading YAML file: {file_path}")
        raise e
    except BoxValueError as e:
        logger.error(f"Error converting YAML to ConfigBox: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e

    return mime.from_file(file_path)

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # noqa
        np.random.seed(seed)
    except Exception:
        pass
