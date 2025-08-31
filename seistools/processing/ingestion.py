import os
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed

from obspy import read
from glob import glob
from multiprocessing import cpu_count


def _check_file(file_path):
    """
    Helper function to check a single file's readability.
    Returns (path, num_waveforms) if readable, otherwise (None, 0).
    """
    KNOWN_EXTENSIONS = {
        "mseed", "miniseed", "sac", "segy", "seg2", "gse", "wav", "y", "bhy",
    }
    ext = os.path.basename(file_path).split(".")[-1].lower()
    if ext not in KNOWN_EXTENSIONS:
        return None, 0
    
    try:
        fmt = ext.upper() if ext!="bhy" else None
        st = read(file_path, headonly=True, format=fmt)
        return file_path, len(st)
    except Exception:
        return None, 0

def scan_files(path_string: str, max_workers:int=8) -> dict:
    """
    Analyzes a given path to determine if it is a file or a folder
    containing ObsPy-readable seismic waveforms.

    Args:
        path_string: The string path to a file or folder.
        max_workers: Number of workers to check for files

    Returns:
        A dictionary with the analysis results.
    """
    results = {
        "is_file": False,
        "is_folder": False,
        "is_obspy_readable": False,
        "total_valid_waveforms": 0,
        "details": "",
        "files": []
    }

    max_workers = cpu_count() if max_workers > cpu_count() else max_workers

    if not os.path.exists(path_string):
        results["details"] = "Path does not exist."
        return results

    if os.path.isfile(path_string):
        results["is_file"] = True
        try:
            st = read(path_string, headonly=True)
            results["is_obspy_readable"] = True
            results["total_valid_waveforms"] = len(st)
            results["details"] = f"File is readable and contains {len(st)} waveform(s)."
            results["files"].append(path_string)
        except Exception as e:
            results["details"] = f"File is not readable by ObsPy: {e}"
    
    elif os.path.isdir(path_string):
        results["is_folder"] = True
        valid_waveforms_count = 0
        valid_files = []

        all_files = glob(f"{path_string}/*")
        all_files = sorted([file for file in all_files if os.path.isfile(file)])
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_check_file, file) for file in all_files]
            
            for future in as_completed(futures):
                file_path, count = future.result()
                if file_path:
                    valid_waveforms_count += count
                    valid_files.append(file_path)

        results["files"] = valid_files
        results["total_valid_waveforms"] = valid_waveforms_count
        results["is_obspy_readable"] = valid_waveforms_count > 0
        results["details"] = (
            f"Folder contains a total of {valid_waveforms_count} "
            f"ObsPy-readable waveform(s) across {len(valid_files)} files."
        )

    return results