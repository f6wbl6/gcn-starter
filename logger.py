# logging/logger.py
import logging


def set_logger():
    logger = logging.getLogger()

    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        streamHandler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s| %(levelname)-5s | %(name)s.%(funcName)s.%(lineno)d | %(message)s"
        )
        streamHandler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)  # Set logger level
        streamHandler.setLevel(logging.INFO)  # Set handler level
        logger.addHandler(streamHandler)

    return logger
