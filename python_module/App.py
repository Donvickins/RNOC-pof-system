"""
Author: Victor Chukwujekwu vwx1423235

This is the programs entry point
"""
import uvicorn
from pathlib import Path
from core.utils.logger_config import LOG_CONFIG

if __name__ == '__main__':
    LOG_DIR = Path("logs")
    uvicorn.run(
        "core.api:app",
        host="0.0.0.0",
        port=5500,
        log_config=LOG_CONFIG
    )