from datetime import datetime

"""
This file contains functions for standardized logging of processes.

Examples:
    src.utils.logger.info('Current reward: 1.45') prints the following log message: 
    "2021-03-01 19:00:00::INFO::Current reward: 1.45"
    
    src.utils.logger.warning('Data not available') prints the following log message: 
    "2021-03-01 19:00:00::WARNING::Data not available"
"""


def _write_message(level, message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}::{level}::{message}", flush=True)


def info(message):
    _write_message(level='INFO', message=message)


def warning(message):
    _write_message(level='WARNING', message=message)


def error(message):
    _write_message(level='ERROR', message=message)
