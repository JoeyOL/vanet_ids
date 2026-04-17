from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import LoggingConfig, PathConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(path_config: PathConfig, logging_config: LoggingConfig) -> logging.Logger:
    path_config.ensure_runtime_dirs()
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config.level))
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path_config.log_file_path, encoding="utf-8")

    if logging_config.json_logs:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if logging_config.log_to_file:
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger("vanet_ids")
