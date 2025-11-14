import sys
from pathlib import Path

logs_dir = Path('logs')
log_file = logs_dir / 'app.log'

if not logs_dir.exists():
    logs_dir.mkdir(parents=True, exist_ok=True)

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "uvicorn": {
            "format": "{levelprefix:<8} {message}",
            "class": "uvicorn.logging.ColourizedFormatter",
            "style": "{",
            "use_colors": True,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "uvicorn",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

if __name__ == "__main__":
    sys.exit(0)