from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import click
import numpy as np
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import Compose
from tqdm import tqdm

from seistools.models import load_eqtransformer, load_munet, load_phasenet
from seistools.processing import (Normalize, ToTensor, scan_files,
                                  single_file_prediction)
from seistools.utils import get_repo_dir


def mlpicker(model_name:str, file_path:str, save_dir:str, 
             freq:int, overlap:float, fmin:float, fmax:float, min_pick_sep:int,
             p_thresh:float=0.5, s_thresh:float=0.5, verbose:bool=False):
    """
    Orchestrates the phase picking process using a machine learning model.

    This is the main function for performing seismic phase picking. It first
    validates the input file path, then initializes the correct model, data
    transformation, and windowing parameters based on the chosen model.
    It utilizes a `ProcessPoolExecutor` to parallelize the phase picking across
    multiple CPU cores, making it efficient for large datasets.

    Args:
        model_name (str): The name of the phase detection model to use.
            Supported models are 'munet', 'eqtransformer', and 'phasenet'.
        file_path (str): The path to a single seismic waveform file or a directory
            containing multiple waveform files.
        save_dir (str): The base directory for saving the output CSV files.
        freq (int): The sampling frequency of the input data in Hz.
        overlap (float): The percentage of overlap between consecutive
            waveform windows.
        fmin (float): The lower frequency for the bandpass filter in Hz.
        fmax (float): The upper frequency for the bandpass filter in Hz.
        min_pick_sep (int): The minimum separation distance in seconds
            between consecutive phase picks.
        p_thresh (float, optional): The probability threshold for P-phase
            detection. Defaults to 0.5.
        s_thresh (float, optional): The probability threshold for S-phase
            detection. Defaults to 0.5.
        verbose (bool, optional): A flag for verbose logging. Defaults to True.

    Returns:
        None: The function does not return any value. It saves the detected
            picks as CSV files in the specified directory.
    """
    
    # Assert the directory/file path is valid and contains waveforms compatible with obspy
    scan = scan_files(file_path)
    logger.info(scan["details"])

    if len(scan["files"]) == 0:
        logger.info("No valid files detected, exiting now")
        return
    
    # Define the window lengths, transformation functions, and load the model weights
    match model_name.lower():
        case "munet":
            win_len = 600
            transform = Normalize("mnmx", axis=0)
            model = load_munet()
        case "eqtransformer":
            win_len = 6000
            transform = Compose([Normalize("mnstd"), ToTensor()])
            model = load_eqtransformer()
        case "phasenet":
            win_len = 3000
            transform = Compose([Normalize("mnstd"), ToTensor()])
            model = load_phasenet()
        case _:
            win_len = 600
            transform = MinMaxScaler()
            model = load_munet()
    
    # Create a wrapper that requires only a valid filepath
    prediction_helper = partial(single_file_prediction, 
                                model_name=model_name,model=model,
                                transform=transform,win_len=win_len,fmin=fmin,
                                fmax=fmax,overlap=overlap,p_thresh=p_thresh,
                                s_thresh=s_thresh,min_pick_sep=min_pick_sep,
                                freq=freq,save_dir=save_dir,verbose=verbose)

    n_cpu = np.floor(cpu_count() / 1.1).astype(int)
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            list(tqdm(executor.map(prediction_helper, scan["files"]),
                      total=len(scan["files"]), position=0,
                      desc="Detecting phase picks",
            ))
    
    return

def validate_frequency_params(ctx, param, value):
    """Validate that fmax doesn't exceed Nyquist frequency"""
    if param.name == 'fmax':
        # Get frequency from context
        freq = ctx.params.get('frequency')
        if freq is not None and value > freq / 2:
            err_msg = f"fmax ({value}) cannot exceed Nyquist frequency ({freq/2:.1f} Hz)"
            raise click.BadParameter(err_msg)
    return value

@click.command()
@click.option("-mdn","--model_name", nargs=1, required=True, help="Pretrained ML picker name",
              type=click.Choice(["munet","eqtransformer","phasenet"], case_sensitive=True))
@click.option("-fp","--file_path", nargs=1, type=click.Path(exists=True, readable=True), 
              required=True, help="Path to input waveforms file / directory")
@click.option("-sdr","--save_dir", nargs=1, type=click.Path(readable=False), 
              default=f"{get_repo_dir()}/picks", help="Output directory to save results. "+\
                "Directory is created if it doesn't exist.")
@click.option("-frq","--frequency", nargs=1, required=True, type=click.FLOAT, 
              help="Sampling frequency of input data in Hz. "+\
                "When using directories, same freq. is applied to all files within the directory.")
@click.option("-ovlp","--overlap", nargs=1, required=True, default=0.25, type=click.FloatRange(0, 1), 
              help="Percentage overlap between consecutive windows (0-1, default: 0.25)")
@click.option("-fmin","--fmin", nargs=1, required=True, type=click.FLOAT, default=1,
              help="Lower frequency for bandpass filter in Hz (default: 1)")
@click.option("-fmax","--fmax", nargs=1, required=True, type=click.FLOAT, 
              callback=validate_frequency_params, default=20,
              help="Upper frequency for bandpass filter in Hz (default: 20)")
@click.option("-mps","--min_pick_sep", nargs=1, required=True, type=click.INT,  default=3,
              help="Minimum separation distance between consecutive picks in seconds (default: 3)")
@click.option("-pt","--p_thresh", nargs=1, type=click.FloatRange(0, 1), default=0.5,
              help="Probability threshold for P-phase detection (default: 0.5)")
@click.option("-st","--s_thresh", nargs=1, type=click.FloatRange(0, 1), default=0.5,
              help="Probability threshold for S-phase detection (default: 0.5)")
@click.option("-v","--verbose", is_flag=True, default=False, help="Enable verbose logging")
def mlpicker_cli(model_name, file_path, save_dir, frequency, overlap, 
                 fmin, fmax, min_pick_sep, p_thresh, s_thresh, verbose):
    """
    Orchestrates seismic phase picking using machine learning models.
    
    Supports munet, eqtransformer, and phasenet models for automated
    P and S phase detection in seismic waveforms.
    """
    mlpicker(
        model_name=model_name,
        file_path=file_path,
        save_dir=save_dir,
        freq=frequency,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        min_pick_sep=min_pick_sep,
        p_thresh=p_thresh,
        s_thresh=s_thresh,
        verbose=verbose
    )