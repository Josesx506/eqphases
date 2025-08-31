import numpy as np
import torch
from scipy.signal import detrend


class Normalize(object):
    """
    Normalize an array either using min-max or mean-std
    valid mode areguments are `mnmx` and `mnstd`
    eps is to prevent zero division in the zscore normalization
    """
    def __init__(self, mode:str="mnstd", axis:int=1, eps:float=1e-8):
        self.mode = mode
        self.eps = eps
        self.axis = axis
        assert self.mode in ["mnmx","mnstd"], "Invalid mode argument"
    
    def _demean(self, x):
        x = x - np.mean(x, axis=self.axis, keepdims=True)
        return x

    def _detrend(self, x):
        x = detrend(x, axis=self.axis)
        return x
    
    def _normalize(self, x):
        if self.mode=="mnmx":
            row_min = x.min(axis=self.axis, keepdims=True)
            row_max = x.max(axis=self.axis, keepdims=True)
            norm = (x-row_min) / ((row_max-row_min) + self.eps)
        elif self.mode=="mnstd":
            row_std = x.std(axis=self.axis, keepdims=True)
            norm = x / (row_std + self.eps)
        return norm

    def __call__(self, sample):
        waveform = sample
        waveform = self._demean(waveform)
        waveform = self._detrend(waveform)
        waveform = self._normalize(waveform)

        return waveform


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,classes=1):
        self.classes = classes

    def __call__(self, sample):
        waveform = sample
        waveform = torch.from_numpy(waveform).to(torch.float32)
        return waveform