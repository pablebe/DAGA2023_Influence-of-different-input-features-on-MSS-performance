#%%

import torch
import scipy.signal# as np


NOLA_COND = scipy.signal.check_NOLA(torch.hann_window(4096),4096,1024)

if NOLA_COND:
    print("NOLA-condition is fullfilled!!")
else:
    print("NOLA-condition is NOT fullfilled!!!")
#%%

FS = 44100

len = 10*FS

x = torch.randn(len)

x_stft = torch.stft(input=x, n_fft=4096, hop_length=1024, win_length=4096, window=torch.hann_window(4096), center=False, normalized=False, onesided=True, pad_mode='reflect', return_complex=False)


x_reconstruct = torch.istft(input=torch.view_as_complex(x_stft), n_fft=4096, hop_length=1024, win_length=4096, window=torch.hann_window(4096), center=True, normalized=False, onesided=True, length=None)

# %%
