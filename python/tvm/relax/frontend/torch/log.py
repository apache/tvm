import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specific name.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        A logger object.
    """
    logger = logging.getLogger(name)
    return logger 