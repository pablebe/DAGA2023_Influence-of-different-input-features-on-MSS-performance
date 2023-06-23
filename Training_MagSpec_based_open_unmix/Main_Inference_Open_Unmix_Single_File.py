#%% Python script to separate vocals from single file with MagSpecLog model from [1].
#   The model is a model trained using the modified Open-Unmix framework.
#
#   References:
#
#   [1] Paul A. Bereuter & Alois Sontacchi, "Influence of Different Input 
#       Features on Musical Source Separation Performance"; Fortschritte der
#       Akustik - DAGA 2023; p.430-433; 
#       URL: https://pub.dega-akustik.de/DAGA_2023/data/daga23_proceedings.pdf
#
#
# Created by Paul A. Bereuter (May 2023)

import toml
import os
import torch
import numpy as np
import soundfile as sf
from openunmix.model import OpenUnmixLog, make_filterbanks
from openunmix.model import OpenUnmix as OpenUnmixMagSpec
from openunmix.transforms import ComplexNorm
from openunmix.filtering import *
config_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(config_path)
config_magspec = toml.load(os.path.join(config_path, 'open_unmix_MagSpec_config.toml'))


log_root = '../trained_models'
FS = 44100
TARGET_STR_SPEC = config_magspec['target']
N_TARGETS_SPEC = config_magspec['n_targets']
NFFT_SPEC = config_magspec['nfft']
NBINS_SPEC = int(NFFT_SPEC/2)+1
NHOP_SPEC = config_magspec['nhop']
HIDDEN_SIZE_SPEC = config_magspec['hidden_size']
BANDWIDTH_SPEC= config_magspec['bandwidth']
NB_CHANNELS_SPEC = config_magspec['nb_channels']
N_LAYERS_SPEC = config_magspec['nb_layers']
UNI_DIR_FLG_SPEC = config_magspec['unidirectional']
MIXTURE_PATH = './audio/mixture.wav'
OUTPUT_PATH = './enhanced_files'

def bandwidth_to_max_bin(rate: float, n_fft: int, bandwidth: float) -> np.ndarray:
    """Convert bandwidth to maximum bin count

    Assuming lapped transforms such as STFT

    Args:
        rate (int): Sample rate
        n_fft (int): FFT length
        bandwidth (float): Target bandwidth in Hz

    Returns:
        np.ndarray: maximum frequency bin
    """
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def apply_noisy_phase(enh_spec:torch.Tensor, mix_stft: torch.Tensor, decoder):
    angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]

    enh_stft = torch.zeros(
        mix_stft.shape, dtype=mix_stft.dtype, device=mix_stft.device
    )
    enh_stft[..., 0] = (enh_spec.unsqueeze(-1).permute(0,3,2,1,4) * torch.cos(angle)).squeeze()
    enh_stft[..., 1] = (enh_spec.unsqueeze(-1).permute(0,3,2,1,4) * torch.sin(angle)).squeeze()
    enh_audio = decoder(enh_stft)
    residual_stft = mix_stft-enh_stft
    residual_audio = decoder(residual_stft)
    estimates_audio = torch.stack((enh_audio,residual_audio),-1)


    return estimates_audio


def main():
    eval_device = torch.device("cpu")
    mixture, fs = sf.read(MIXTURE_PATH)
    song_name = MIXTURE_PATH.split('/')[-2]
    song_name = '_'.join(song_name.split(' '))
    mixture = torch.Tensor(mixture.T).unsqueeze(0)
    audio_len = mixture.size(-1)

    comp_norm = ComplexNorm(mono= NB_CHANNELS_SPEC == 1)


    MAX_BIN_SPEC = bandwidth_to_max_bin(FS, NFFT_SPEC, BANDWIDTH_SPEC)
    MagSpecLog_wo_LR_sched_w_rand_mix_MSE = OpenUnmixLog(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)
    MagSpec_w_LR_sched_w_rand_mix_MSE = OpenUnmixMagSpec(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)

    chosen_epoch_MagSpecLog_wo_LR_sched_w_rand_mix_MSE = 879
    CKPT_PATH_MagSpecLog_wo_LR_sched_w_rand_mix_MSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/MagSpecLog_MSE_w_LR_schedule/Open_Unmix_model_'+str(chosen_epoch_MagSpecLog_wo_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_MagSpecLog_wo_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_MagSpecLog_wo_LR_sched_w_rand_mix_MSE, map_location=eval_device)
    MagSpecLog_wo_LR_sched_w_rand_mix_MSE.load_state_dict(state_dict_MagSpecLog_wo_LR_sched_w_rand_mix_MSE)
    MagSpecLog_wo_LR_sched_w_rand_mix_MSE.eval()

    stft, istft = make_filterbanks(n_fft=NFFT_SPEC, n_hop=NHOP_SPEC, sample_rate=FS, center=True, length = audio_len)
    mix_stft = stft(mixture)
    mix_spec = comp_norm(mix_stft)[0].unsqueeze(0)

    est_vocals_spec = MagSpecLog_wo_LR_sched_w_rand_mix_MSE(mix_spec)

    est_vocals_audio = apply_noisy_phase(est_vocals_spec, mix_stft,istft).detach()

    sf.write(os.path.join(OUTPUT_PATH,song_name+'_separated_vocals.wav'),est_vocals_audio[...,0].squeeze().t(), samplerate = FS)

    cut_start = int(170*fs)
    cut_stop = int(176*fs)
    
    sf.write(os.path.join(OUTPUT_PATH,song_name+'_separated_vocals_cut.wav'),est_vocals_audio[...,cut_start:cut_stop,0].squeeze().t(), samplerate = FS)

if __name__ == "__main__":
    main()