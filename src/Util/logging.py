import logging


def logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log", encoding='utf-8'),  # Note the encoding parameter
            logging.StreamHandler()
        ]
    )