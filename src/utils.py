import logging


def setup_logging(file_name="logfile.log"):
    """Set up logging"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=file_name,
        filemode="w",
    )
    return logging
