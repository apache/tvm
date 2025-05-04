"""Simple logging module for TVM Relax PyTorch frontend."""
import logging


def get_logger(name):
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