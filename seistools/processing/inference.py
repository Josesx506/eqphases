import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from detecta import detect_peaks
from loguru import logger
from obspy import UTCDateTime

from seistools.processing.windowing import WindowWaveforms
from seistools.utils import get_repo_dir


def detect_phases(model_name:str, model, 
                  time_list:list, id_list:list, arr_list:np.ndarray|torch.Tensor,
                  p_thresh:float, s_thresh:float, min_pick_sep:int=3, freq:int=100,
                  save_dir:str=f"{get_repo_dir()}/picks", verbose:bool=False):
    """
    Detects seismic P and S phases using a pre-trained machine learning model.

    This function processes a batch of seismic waveform data using a specified
    model (MuNet, EQTransformer, or PhaseNet) to predict P- and S-phase
    probabilities. It then applies peak detection with configurable thresholds
    and a minimum separation distance to identify individual phase picks.
    The detected picks, along with their associated station IDs and prediction
    probabilities, are compiled into a pandas DataFrame and saved to a CSV file.

    Args:
        model_name (str): The name of the phase detection model. Must be one of
            'munet', 'eqtransformer', or 'phasenet'.
        model: The pre-trained machine learning model object. The type depends
            on the `model_name`. munet uses tensorflow. eqtransformer and phasenet 
            are retrieved as pytorch objects.
        time_list (list): A list of ObsPy `UTCDateTime` objects representing
            the start time of each waveform in `arr_list`.
        id_list (list): A list of strings containing the station IDs for each
            waveform.
        arr_list (np.ndarray | torch.Tensor): The seismic waveform data as a
            NumPy array or PyTorch tensor.
        p_thresh (float): The probability threshold for detecting P-phases.
            Peaks with a probability below this value will be ignored.
        s_thresh (float): The probability threshold for detecting S-phases.
            Peaks with a probability below this value will be ignored.
        min_pick_sep (int, optional): The minimum separation distance in seconds
            between consecutive phase picks. Defaults to 3.
        freq (int, optional): The sampling frequency of the input data in Hz.
            Defaults to 100.
        save_dir (str, optional): The base directory where the output CSV files
            will be saved. A subdirectory for the specific model and date will be
            created within this directory. Defaults to the repository's picks folder.
        verbose (bool, optional): A flag to enable or disable verbose output
            during model prediction. Defaults to False.

    Raises:
        ValueError: If `model_name` is not one of the supported models.

    Returns:
        bool: The function returns a boolean indicating whether the process was completed. 
            It saves the detected picks as a CSV file in the specified directory.
    """
    if verbose:
        logger.info(f"Detecting phases using {model_name}")
    err_msg = "Invalid model name. Should be one of [`munet`,`eqtransformer`,`phasenet`]"
    assert model_name in ["munet","eqtransformer","phasenet"], err_msg

    match model_name:
        case "munet":
            e,p,s = model.predict(arr_list, verbose=verbose)
            e,p,s = e.squeeze(-1),p.squeeze(-1),s.squeeze(-1)
        case "eqtransformer":
            with torch.no_grad():
                e,p,s = model(arr_list.to(model.device))
                e,p,s = e.detach().numpy(),p.detach().numpy(),s.detach().numpy()
        case "phasenet":
            with torch.no_grad():
                pred = model(arr_list.to(model.device))
                pred = pred.detach().numpy()
                e,p,s = pred[:,0,:],pred[:,1,:],pred[:,2,:]
        case _:
            raise ValueError("Invalid model name")
    
    # Lists to save the results
    st_id,phase,prob,times_ = [],[],[],[]

    mpd = min_pick_sep * freq # Minimum pick distance
    for i, st_time in enumerate(time_list):
        ids,_e,_p,_s = id_list[i], e[i], p[i], s[i]
        """
        When the  p- or s- probabilities are above the threshold, trigger a detection
        """
        # phase pick probability indices
        p_pb_idx = detect_peaks(_p, mph=p_thresh, mpd=mpd, show=False)
        s_pb_idx = detect_peaks(_s, mph=s_thresh, mpd=mpd, show=False)

        if len(p_pb_idx) > 0:
            for pIdx in p_pb_idx:
                # Append the Station ID
                st_id.append(ids)
                phase.append("P")
                prob.append(np.round(_p[pIdx], 2))
                times_.append(UTCDateTime(st_time.timestamp + (pIdx / 100)))

        if len(s_pb_idx) > 0:
            for sIdx in s_pb_idx:
                # Append the Station ID
                st_id.append(ids)
                phase.append("S")
                prob.append(np.round(_s[sIdx], 2))
                times_.append(UTCDateTime(st_time.timestamp + (sIdx / 100)))

    # Create a dataframe for the output
    df_picks = pd.DataFrame(
        {
            "Network-Station": st_id,
            "Phase": phase,
            "Prob": prob,
            "Time": times_,
        }
    )

    # Format the dataframe time column
    df_picks["Time"] = df_picks["Time"].apply(lambda x: str(x)[:-5])
    df_picks["Time"] = pd.to_datetime(df_picks["Time"])
    df_picks = df_picks.drop_duplicates()
    df_picks = df_picks.sort_values(by=["Time","Network-Station","Phase"]).reset_index(drop=True)

    date = UTCDateTime(time_list[0])
    yr, mo, dy = date.year, str(date.month).zfill(2), str(date.day).zfill(2)
    
    if df_picks.empty:
        if verbose:
            logger.info(f"No events detected for {ids} on {yr}-{mo}-{dy}.")
        return False
    else:
        out_dir = f"{save_dir}/{yr}{mo}{dy}_{model_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        out_name = f"{out_dir}/{ids}_detections.csv"
        df_picks.to_csv(out_name, index=False)
    
    if verbose:
        logger.info(
            f"Finished event detections for station {ids} on {yr}-{mo}-{dy}."\
                +f" csv file saved in {out_dir}"
        )

    return True


def single_file_prediction(file_path:str, fmin:float, fmax:float, win_len:float, transform, 
                           model, model_name:str, overlap:float, p_thresh:float, 
                           s_thresh:float, min_pick_sep:int=3, freq:int=100, 
                           save_dir:str=f"{get_repo_dir()}/picks", verbose:bool=False):
    """
    Performs phase picking on a single seismic waveform file.

    This function serves as a wrapper to orchestrate the entire phase detection
    workflow for a single file. It reads the waveform, applies a bandpass filter,
    windows the data into smaller segments with a specified overlap, and then
    calls `detect_phases` to perform the actual phase picking using a
    machine learning model. The results are saved to a CSV file.

    Args:
        file_path (str): The path to the seismic waveform file.
        fmin (float): The lower frequency for the bandpass filter in Hz.
        fmax (float): The upper frequency for the bandpass filter in Hz.
        win_len (int): The length of the waveform windows in samples.
            Should align with the defined input samples for the ML model name.
        transform: A function or method to transform the waveform data before
            inputting it into the model.
        model: The pre-trained machine learning model object.
        model_name (str): The name of the phase detection model.
        overlap (float): The percentage of overlap between consecutive
            waveform windows. Ranges between 0-1.
        p_thresh (float): The probability threshold for detecting P-phases.
        s_thresh (float): The probability threshold for detecting S-phases.
        min_pick_sep (int, optional): The minimum separation distance in seconds
            between consecutive phase picks. Defaults to 3.
        freq (int, optional): The sampling frequency of the input data in Hz.
            Defaults to 100.
        save_dir (str, optional): The base directory for saving the output
            CSV files. Defaults to the repository's picks folder.
        verbose (bool, optional): A flag for verbose logging. Defaults to False.

    Returns:
        None: The function does not return any value. It saves the results
            via the `detect_phases` function.
    """
    
    waveform = WindowWaveforms(file_path,freq=freq,overlap=overlap)
    waveform.resample_waveform(fmin=fmin,fmax=fmax)
    time_list, id_list, arr_list = waveform.window_timeseries(transform,win_len)

    # Perform event detection
    complete = detect_phases(model_name=model_name, model=model,
                             time_list=time_list, id_list=id_list,
                             arr_list=arr_list, p_thresh=p_thresh,
                             s_thresh=s_thresh, min_pick_sep=min_pick_sep,
                             freq=freq, save_dir=save_dir, verbose=verbose)

    if complete and verbose:
        logger.info("Phase picking completed successfully.")