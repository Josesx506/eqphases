"""
Scripts and functions for detecting phases with ML models
"""

from seistools.processing.dataloaders import Normalize, ToTensor
from seistools.processing.inference import (detect_phases,
                                            single_file_prediction)
from seistools.processing.ingestion import scan_files
from seistools.processing.windowing import WindowWaveforms
