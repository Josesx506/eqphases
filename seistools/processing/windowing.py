import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import obspy
import torch
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import Compose


class WindowWaveforms:
    def __init__(self, file_path:str, 
                 freq:int=100, overlap:float=0.25, 
                 read:bool=True, stream:obspy.Stream=None,
                 ) -> None:
        
        if read == False and stream is not None:
            self.stream = stream
        else:
            self.stream = obspy.read(file_path)
            self.stream = self.stream.merge()
        self.freq = freq
        self.overlap = overlap
        self.stream_id = self.stream[0].get_id()
        self.stream_id = ".".join(self.stream_id.split(".")[:2])


    def resample_waveform(self, fmin:float=1, fmax:float=20):
        """
        Resample the waveforms to match the input frequencies for the models.
        Data is also bandpass filtered by specified limites

        Args:
            fmin (int, optional): frequency lower limit. Defaults to 1.
            fmax (int, optional): frequency upper limit. Defaults to 20.

        Returns:
            None
        """
        self.begin_time = min([st.stats.starttime for st in self.stream])
        self.end_time = max([st.stats.endtime for st in self.stream])

        self.stream = self.stream.detrend("demean")
        self.stream = self.stream.filter("bandpass", freqmin=fmin, freqmax=fmax)
        self.stream.interpolate(sampling_rate=self.freq, method="linear")

        self.stream = self.stream.trim(
            self.begin_time, self.end_time, pad=True, fill_value=0
        )

    def window_timeseries(self, transforms=MinMaxScaler(), window=600):
        """
        Crop windows of daily waveforms and normalize each window

        Args:
            stream (_type_): obspy.Stream
            transforms (object): Can be min-max scaler for numpy arrays or 
            window (int, optional): Window length (samples). Defaults to 600.

        Returns:
            (tuple): (time_list, id_list, arr_list)
            time_list: list of the start times of each window
            id_list: list of each station id
            arr_list: list of windowed waveforms as arrays
        """
        shift = int(window * (1 - self.overlap))
        nsta = len(self.stream)
        if nsta <= 0:
            raise ValueError("Waveforms should have at least one channel")
        if nsta > 3:
            raise ValueError("Maximum number of stations should not exceed 3")

        # maximum number of samples in the dataset
        ns = max(len(tr.data) for tr in self.stream)

        # Pad any channel where the beginning or end of channel data is missing. Does not fill gaps in-between
        data_pad = []
        for tr in self.stream:
            pad_front = np.ceil(np.abs(self.begin_time - tr.stats.starttime) * self.freq).astype(int)
            pad_back = np.ceil(np.abs(tr.stats.endtime - self.end_time) * self.freq).astype(int)
            padded = np.pad(tr.data,(pad_front + shift, pad_back + shift),mode="mean")
            data_pad.append(padded)
        min_pad = min([len(arr) for arr in data_pad])  # Remove excess padding from ceil
        data_pad = np.stack([arr[:min_pad] for arr in data_pad], axis=1)

        # Slice the data windows
        times = np.arange(0, ns, shift, dtype="int")
        window_data = []
        for i in range(1, len(times)):
            sliced_data = data_pad[i * shift : i * shift + window, :nsta]
            # Pad. array with zeros when number of stations are insufficient
            if nsta < 3:
                tmp_data = np.zeros((window, 3))
                tmp_data[:, :nsta] = sliced_data
                sliced_data = tmp_data
            
            # Normalize the data
            if isinstance(transforms, Compose):
                sliced_data = transforms(sliced_data.T)
            elif isinstance(transforms, MinMaxScaler):
                sliced_data = transforms.fit_transform(sliced_data)
            else:
                sliced_data = transforms(sliced_data)
            window_data.append(sliced_data)

        if isinstance(transforms, Compose):
            arr_list = torch.stack(window_data, dim=0)
        else:
            arr_list = np.stack(window_data, axis=0)
        
        time_list = [self.begin_time + t / self.freq for t in times][:-1]
        id_list = [self.stream_id for n in time_list]

        return time_list, id_list, arr_list