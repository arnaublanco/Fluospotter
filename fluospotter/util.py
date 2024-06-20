"""Utility helper functions."""

from typing import Callable, Iterable, Tuple, Union
import importlib
import os
import random

from PIL import Image
from PIL import UnidentifiedImageError
from PIL.TiffTags import TAGS
import numpy as np
import pandas as pd
import psutil
import threading
import time
import sys


def monitor_ram_usage(stop_event):
    # Get the virtual memory details
    mem = psutil.virtual_memory()
    total_ram_mb = mem.total / (1024 * 1024)  # Convert bytes to MB
    threshold_mb = 0.98 * total_ram_mb

    while not stop_event.is_set():
        # Get the current process
        process = psutil.Process(os.getpid())

        # Get the memory info
        mem_info = process.memory_info()

        # Convert bytes to MB
        vms_mb = mem_info.vms / (1024 * 1024)  # Virtual Memory Size

        # Check if the RAM usage exceeds the threshold
        if vms_mb > threshold_mb:
            print(f"Memory RAM exceeded {threshold_mb:.2f} MB, stopping the program.")
            sys.exit(1)

        # Sleep for a while before checking again
        time.sleep(1)
