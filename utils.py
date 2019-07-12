import logging
import sys


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


if __name__ == "__main__":
    for each_path in sys.path:
        print(each_path)

    print("hello world")
