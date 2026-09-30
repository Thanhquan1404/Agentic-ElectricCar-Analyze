import logging
import sys
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Colorized formatter for terminal output."""

    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMAT = (
        "%(asctime)s | "
        "%(levelname)-8s | "
        "%(name)s | "
        "%(filename)s:%(lineno)d | "
        "%(funcName)s() | "
        "%(message)s"
    )

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with level-specific terminal colors."""
        log_format = self.FORMATS.get(record.levelno, self.FORMAT)

        formatter = logging.Formatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        return formatter.format(record)


class DetailedFormatter(logging.Formatter):
    """Detailed formatter for persistent log files."""

    FORMAT = (
        "%(asctime)s | "
        "%(levelname)-8s | "
        "PID=%(process)d | "
        "TID=%(thread)d | "
        "%(name)s | "
        "%(filename)s:%(lineno)d | "
        "%(funcName)s() | "
        "%(message)s"
    )

    def __init__(self) -> None:
        super().__init__(
            fmt=self.FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logger(
    name: str = "AI_Agent",
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialize and configure the application logger.

    Args:
        name: Logger name used to identify the application or module.
        log_file: Path used to persist log messages. Set to None to disable
            file logging.
        level: Minimum logging level.

    Returns:
        Configured logging.Logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers when setup_logger is called multiple times.
    if logger.handlers:
        return logger

    # Prevent log messages from being propagated to the root logger.
    logger.propagate = False

    # Console handler for real-time terminal monitoring.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # File handler for persistent application logs.
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(DetailedFormatter())

        logger.addHandler(file_handler)

    return logger


# Default logger for direct import and reuse across the application.
logger = setup_logger()
