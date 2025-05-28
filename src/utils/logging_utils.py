import logging
from typing import Optional
from enum import Enum

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

def setup_logger(name: str, level: LogLevel = LogLevel.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.value)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_mask_validation(logger: logging.Logger, 
                       mask_shape: tuple,
                       expected_shape: tuple,
                       is_test_run: bool = False) -> None:
    """
    Log mask validation results with appropriate context.
    
    Args:
        logger: Logger instance
        mask_shape: Actual shape of the mask
        expected_shape: Expected shape of the mask
        is_test_run: Whether this is a test run
    """
    run_type = "TEST" if is_test_run else "EXPERIMENT"
    logger.info(f"[{run_type}] Validating mask shape: {mask_shape} (expected: {expected_shape})")
    
    if mask_shape != expected_shape:
        logger.error(
            f"[{run_type}] Mask shape mismatch! Expected {expected_shape}, got {mask_shape}"
        )
    else:
        logger.info(f"[{run_type}] Mask shape validation passed") 