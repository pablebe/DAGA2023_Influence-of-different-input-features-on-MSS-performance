#%%
import torch
import soundfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from openunmix_slicq.CQT_class import CQT, VCQT
from openunmix_slicq.transforms import make_filterbanks, ComplexNorm, NSGTBase
from openunmix_slicq.nsgt_orig import NSGT, NSGT_sliced, LogScale, LinScale, MelScale, OctScale, VQLogScale, BarkScale, SndReader
from openunmix_slicq.nsgt_orig.slicq import overlap_add_slicq
#%%
FS = 44100


train_device = torch.device("cpu")

t_start = 91 #s=1:31
t_stop = 101 #s=1:41

WINLEN = 4096
HOPSIZE = 1204

test_sample_path = '/home/bereuter/MSS/open_unmix_experiments/03_Experiments/NSGT_based_open_unmix/Test_audio/vocals.wav'

sig,fs = soundfile.read(test_sample_path,start=int(t_start*FS),stop=int(t_stop*FS))
#sig,fs = soundfile.read(test_sample_path)

sig = torch.Tensor(sig.T)
#sig = sig.unsqueeze(0)
sig = sig.unsqueeze(0)#.unsqueeze(0)
sig_len = sig.size(-1)
#%%

# nsgt_base = NSGTBase(scale='cqlog',
#     fbins=12,
#     fmin=32,
#     fmax=FS/2,
#     fs=FS,
#     matrixform= True,
#     device=train_device,
# #    sllen = WINLEN,
# #    gamma = 3,
# #    trlen = HOPSIZE,
# )

# nsgt, insgt = make_filterbanks(
#     nsgt_base, sample_rate=FS
# )
#n_octs = 4
f_min = 100
n_bins_per_octave = 12*6
f_max = FS/2#f_min*2**n_octs

# calculate how many 
n_bnds = int(np.ceil(np.log2(f_max/f_min)*n_bins_per_octave))

# scl = VQLogScale(fmin=f_min,fmax=f_max,bnds=n_bnds,gamma=30)

# if True:
#     sllen, trlen = scl.suggested_sllen_trlen(fs)
# else:
#     sllen = WINLEN
#     trlen = HOPSIZE

# slicq = NSGT(scl,fs,sig_len, 
#                     real=True,
#                     matrixform=True,
#                     multichannel=False,
#                     device="cpu"
#                     )

vcqt_gamma_0 = VCQT(f_min, f_max, n_bnds, Qvar=1.66, fs=fs, audio_len=sig_len, multichannel=True, device = "cpu", split_0_nyq=False)
vcqt_gamma_30 = VCQT(f_min, f_max, n_bnds, Qvar=1.66, fs=fs, audio_len=sig_len , multichannel=True, device = "cpu", split_0_nyq=False)

#cqt = CQT(f_min, f_max, n_bnds, fs=fs, audio_len=sig_len, device="cpu", split_0_nyq=False)
#sig_CQ = librosa.cqt(sig.squeeze().numpy(), sr = FS, fmin=32, bins_per_octave=12)
#sig_reconst = torch.Tensor(librosa.icqt(sig_CQ, sr = FS, fmin=32, bins_per_octave=12, length = sig_len))

encoder = lambda x: vcqt_gamma_0.fwd(x)
rand = torch.rand((36,2,441000))
rand[3,:] = sig
c0= encoder(rand)#nsgt(sig)#[0]
#sig_CQ_abs = (sig_VCQ_0.squeeze()**2).sum(-1).sqrt()

#    for jj in range(int(n_bins_per_octave/12)):
#        semitone_ids[jj,ii+1] = jj+1+ii
#sig_VCQ_30 = vcqt_gamma_30.fwd(sig)

#sig_CQ = cqt.fwd(sig)
sig_reconst = vcqt_gamma_0.bwd(c0)[3,:]#insgt(sig_CQ,sig_len)

reconst_error= torch.mean((sig.squeeze()-sig_reconst.squeeze())**2)
print('Reconstruction-Error: '+str(reconst_error))

soundfile.write('/home/bereuter/MSS/open_unmix_experiments/03_Experiments/NSGT_based_open_unmix/Test_audio/vocals_cut.wav',sig.squeeze().T,samplerate=fs)
soundfile.write('/home/bereuter/MSS/open_unmix_experiments/03_Experiments/NSGT_based_open_unmix/Test_audio/vocals_reconst_sliCQ.wav',sig_reconst.squeeze().T,samplerate=fs)

print("stop")

# %%
#mls = sig_CQ.reshape((1,1,97,232*217))
#mls_cq = 20*torch.log10((sig_CQ.squeeze()**2).sum(0).sqrt())
mls_vcq_30 = 20*torch.log10((c0.squeeze()**2).sum(-1).sqrt())[3,:]
#mls_vcq_0 = 20*torch.log10((sig_VCQ_0.squeeze()**2).sum(-1).sqrt())

# chop = sig_CQ.shape[-1]
# mls = mls[:, :, :, int(chop/2):]
# mls = mls[:, :, :, :-int(chop/2)]
#plt.pcolormesh(mls_cq.T)

plt.figure()
plt.pcolormesh(mls_vcq_30[0,:])
plt.figure()
plt.pcolormesh(mls_vcq_30[1,:])
# plt.figure()
# plt.pcolormesh(mls_vcq_0[0,:])
print("stop")
# %%
