import os
import sys
from loguru import logger

# Define log directory and file
log_dir = os.path.join(os.path.dirname(__file__), "logs")
log_file = os.path.join(log_dir, "app.log")
os.makedirs(log_dir, exist_ok=True)

# Remove default handler to avoid duplicate logs
logger.remove()

# Add a new handler to log to a file
logger.add(
    log_file,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

logger.info("welcome to VideoCaptioning!")