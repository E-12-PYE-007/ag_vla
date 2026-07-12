import logging
from datetime import datetime
from pathlib import Path

class TrainingLogger:
    def __init__(self):
        self._logger = logging.getLogger("training")

    #TODO: instead of datetime, name by run metadata
    def setup(self, log_dir: Path, filename: str) -> None:
        # Extract components before and after the dot
        stem, _, ext = filename.rpartition(".")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{stem}_{timestamp}.{ext}"

        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Handler to write into file
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        # Handler to write into terminal
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)

        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)
        self.info(f"Log file: {log_path}")

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warn(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

logger = TrainingLogger()