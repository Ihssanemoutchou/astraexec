import logging
from pathlib import Path
from datetime import datetime


class Logger:
    """
    AstraExec Logger

    Journalise les exécutions des outils.
    """

    def __init__(self, log_dir="logs"):

        Path(log_dir).mkdir(exist_ok=True)

        self.logger = logging.getLogger("AstraExec")

        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:

            formatter = logging.Formatter(

                "%(asctime)s | %(levelname)s | %(message)s"

            )

            file_handler = logging.FileHandler(

                Path(log_dir) / "astra_exec.log",

                encoding="utf-8"

            )

            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    # =====================================================
    # Succès
    # =====================================================

    def log_success(

        self,

        tool,

        execution_time,

    ):

        self.logger.info(

            f"SUCCESS | Tool={tool} | Time={execution_time:.4f}s"

        )

    # =====================================================
    # Erreur
    # =====================================================

    def log_error(

        self,

        message,

        execution_time=0,

    ):

        self.logger.error(

            f"ERROR | Time={execution_time:.4f}s | {message}"

        )

    # =====================================================
    # Information
    # =====================================================

    def log_info(self, message):

        self.logger.info(message)

    # =====================================================
    # Warning
    # =====================================================

    def log_warning(self, message):

        self.logger.warning(message)

    # =====================================================
    # Début d'action
    # =====================================================

    def log_action_start(

        self,

        tool,

    ):

        self.logger.info(

            f"START | {tool}"

        )

    # =====================================================
    # Fin d'action
    # =====================================================

    def log_action_end(

        self,

        tool,

    ):

        self.logger.info(

            f"END | {tool}"

        )

    # =====================================================
    # Event personnalisé
    # =====================================================

    def log_event(

        self,

        event,

        details="",

    ):

        self.logger.info(

            f"EVENT | {event} | {details}"

        )

    # =====================================================
    # Timestamp
    # =====================================================

    def now(self):

        return datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        )


if __name__ == "__main__":

    logger = Logger()

    logger.log_action_start("FusionSearch")

    logger.log_success(

        "FusionSearch",

        0.2415

    )

    logger.log_warning(

        "Chunk faible détecté."

    )

    logger.log_error(

        "Document introuvable.",

        0.04

    )

    logger.log_action_end(

        "FusionSearch"

    )

    logger.log_event(

        "TEST",

        "Logger opérationnel."

    )