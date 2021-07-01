import logging
import os

REPO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
DATA_DIR = os.path.join(REPO_ROOT_DIR, 'data')


def setup_logging():
    logging.basicConfig(
        format='{asctime} [{levelname}] ({name}): {message}',
        datefmt='%Y-%m-%d %H:%M:%S',
        style='{',
        level=logging.INFO
    )
