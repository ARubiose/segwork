"""Loggers. 

Use the loggers to:
    *   Display console output for ordinary usage
    *   Report events that occur during normal operation of a framework
    **  Display and save training logs
"""
from abc import abstractmethod
import logging

class Logger:
    """Abstract base class to save logs"""
    pass

    @abstractmethod
    def set_logger(self, *args, **kwargs):
        """Set up logger configuration"""
        pass

class LocalLogger(Logger):
    """Class to save training logs by console and documents"""

    def add_handler():
        pass

    def set_logger( 
        self,
        log_path: str = 'log.log',
        format: str = "%(levelname)-8s %(message)s",
        level: int = logging.DEBUG
        ) -> None:
        """Set up basic configuration and format for logger"""
        logging.basicConfig(filename=log_path, format=format, level=level)



    
        

