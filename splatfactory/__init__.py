import logging
import os


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a rank-aware logger.

    Args:
        name: Logger name (typically __name__)
        level: Optional log level override for all ranks. If None, uses INFO on rank 0,
               WARNING on other ranks.

    Returns logger with INFO level on rank 0, WARNING on other ranks.
    Drop-in replacement for logging.getLogger(__name__).
    """
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(level)
    else:
        rank_str = os.environ.get("RANK")
        if rank_str is None or int(rank_str) == 0:
            log.setLevel(logging.INFO)
        else:
            log.setLevel(logging.WARNING)
    return log


formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = get_logger(__name__)
logger.addHandler(handler)
logger.propagate = False

__module_name__ = __name__
