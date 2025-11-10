import sys
from pathlib import Path
from loguru import logger

# Define the base directory (where the current file is located)
base_dir = Path(__file__).resolve().parent

# Define log directory and log file path
log_dir = base_dir / "logs"
log_file = log_dir / "app.log"

# Create directory if missing
log_dir.mkdir(parents=True, exist_ok=True)

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