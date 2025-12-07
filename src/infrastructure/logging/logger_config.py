"""
Logging Configuration for RAG Application

This module sets up comprehensive logging for tracking:
- User actions
- System operations
- Inter-service communication
- Performance metrics
- Errors and exceptions
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


class RAGLogger:
    """
    Centralized logging configuration for the RAG application.

    Logs are written to:
    - logs/rag_app.log - All application logs
    - logs/user_actions.log - User-specific actions
    - logs/chromadb.log - ChromaDB operations
    - logs/openai.log - OpenAI API calls
    - logs/errors.log - Errors only
    """

    _initialized = False

    @classmethod
    def setup_logging(cls, log_dir: str = "logs"):
        """
        Set up logging configuration.

        Args:
            log_dir: Directory to store log files
        """
        if cls._initialized:
            return

        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Define log format with more details
        detailed_format = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Main application logger
        main_logger = logging.getLogger('rag_app')
        main_logger.setLevel(logging.INFO)
        main_handler = RotatingFileHandler(
            log_path / 'rag_app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_handler.setFormatter(detailed_format)
        main_logger.addHandler(main_handler)

        # 2. User actions logger
        user_logger = logging.getLogger('user_actions')
        user_logger.setLevel(logging.INFO)
        user_handler = RotatingFileHandler(
            log_path / 'user_actions.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        user_handler.setFormatter(simple_format)
        user_logger.addHandler(user_handler)

        # 3. ChromaDB operations logger
        chroma_logger = logging.getLogger('chromadb_ops')
        chroma_logger.setLevel(logging.DEBUG)
        chroma_handler = RotatingFileHandler(
            log_path / 'chromadb.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        chroma_handler.setFormatter(detailed_format)
        chroma_logger.addHandler(chroma_handler)

        # 4. OpenAI API logger
        openai_logger = logging.getLogger('openai_api')
        openai_logger.setLevel(logging.INFO)
        openai_handler = RotatingFileHandler(
            log_path / 'openai.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        openai_handler.setFormatter(detailed_format)
        openai_logger.addHandler(openai_handler)

        # 5. Error logger
        error_logger = logging.getLogger('errors')
        error_logger.setLevel(logging.ERROR)
        error_handler = RotatingFileHandler(
            log_path / 'errors.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        error_handler.setFormatter(detailed_format)
        error_logger.addHandler(error_handler)

        # Also add console handler for main logger (optional)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_handler.setFormatter(simple_format)
        main_logger.addHandler(console_handler)

        cls._initialized = True
        main_logger.info("=" * 80)
        main_logger.info(f"RAG Application Logging Initialized - {datetime.now()}")
        main_logger.info("=" * 80)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger by name.

        Args:
            name: Logger name (rag_app, user_actions, chromadb_ops, openai_api, errors)

        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)


# Initialize logging on module import
RAGLogger.setup_logging()
