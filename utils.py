import logging
import sys
from datetime import datetime
from pathlib import Path


def build_logger() -> logging.Logger:
    # Init logger
    logger = logging.getLogger(name=__file__)
    logger.setLevel(level=logging.INFO)

    # Create handler
    cmd_handler = logging.StreamHandler()
    cmd_handler.setLevel(logging.INFO)

    # Create formatter and add to handler
    cmd_formater = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cmd_handler.setFormatter(cmd_formater)

    # Set handler
    logger.addHandler(cmd_handler)

    return logger


def get_log_dir(ds_type: str, model_type: str) -> Path:
    current = datetime.now().strftime('%b%d_%H-%M-%S')
    dirname = '-'.join([model_type, current])
    base_path = Path('./runs')
    log_path = base_path / ds_type / dirname
    return log_path


if __name__ == "__main__":
    print(get_log_dir(ds_type='kaggle', model_type='hello'))
