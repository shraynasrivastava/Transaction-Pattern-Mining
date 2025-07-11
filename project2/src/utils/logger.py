import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(logging_config: dict):
    """Set up logging configuration from config dictionary.
    
    Args:
        logging_config: Dictionary containing logging configuration
    """
    # Ensure log directory exists
    log_file = logging_config['log_file']
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=logging_config.get('max_bytes', 10485760),  # 10MB default
                backupCount=logging_config.get('backup_count', 5)
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def setup_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """Set up a logger with both file and console handlers (legacy function).
    
    Args:
        name: Name of the logger
        log_file: Path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Check if logger already has handlers to avoid duplicates
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (if log_file specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger 