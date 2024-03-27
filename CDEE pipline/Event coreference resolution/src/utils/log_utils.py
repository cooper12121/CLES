import datetime
import logging

from utils.io_utils import create_and_get_path


def create_logger_with_fh(params_str=""):
    log_file = params_str + ".log"

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
