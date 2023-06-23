import museval
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def get_BSS_metrics(targets: Tensor, estimates: Tensor, num_workers: int = 1, OUTPUT_SCORE_PATH: str=None)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    batch_size = targets.size(dim=0)
    scores = Parallel(n_jobs = num_workers)(delayed(calc_metrics)(targets[ii,:].detach().cpu(), estimates[ii,:].detach().cpu(), OUTPUT_SCORE_PATH)
                                         for ii in range(batch_size))

    return torch.nanmean(torch.Tensor(np.array(scores)), dim=0).squeeze(-1)


def calc_metrics(targets: Tensor, estimates: Tensor, OUTPUT_SCORE_PATH: str):
    n_sources = targets.size(0)
    if sum(sum(targets.sum(1)==torch.zeros(2,2)))<1:
        #only compute metrics if the target signal on both channels (L&R) is non-zero for all sources
#        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(targets,estimates,np.inf)
        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(targets,estimates,window=int(1*44100),hop=int(1*44100))
    else:
        nan_array = np.empty((n_sources,1))
        nan_array[:] = np.array(float('nan'))
        sdr = nan_array #float('nan')
        isr = nan_array #float('nan')
        sir = nan_array #float('nan')
        sar = nan_array #float('nan')

#    return sdr, isr, sir, sar
    return np.nanmedian(sdr,1), np.nanmedian(isr,1), np.nanmedian(sir,1), np.nanmedian(sar,1)
