import sys
from datetime import datetime

def _log(level, message, color_code=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reset = "\033[0m"
    print(f"{color_code}[{timestamp}] [{level}] {message}{reset}", flush=True)

def info(message):
    _log("INFO", message, "\033[32m")  # Green

def warn(message):
    _log("WARN", message, "\033[33m")  # Yellow

def error(message):
    _log("ERROR", message, "\033[31m")  # Red

def debug(message):
    _log("DEBUG", message, "\033[36m")  # Cyan
