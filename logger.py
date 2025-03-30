import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("agentic_system")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("logs/system.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def get_logger(name=None):
    if name:
        return logging.getLogger(f"agentic_system.{name}")
    return logger