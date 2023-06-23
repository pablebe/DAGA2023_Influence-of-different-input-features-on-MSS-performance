#%% Python script to evaluate modified Open-Unmix models [1]. 
#   Different modifications on the framework are compared in [2] using the MUSDB18 dataset [3].
#   
#
#   References:
#
#   [1] Fabian-Robert Stöter, Stefan Uhlich, Antoine Liutkus & Yuki Mitsufuji
#       "Open-Unmix - A Reference Implementation for Music"; Sept. 2019;
#       URL: https://www.theoj.org/joss-papers/joss.01667/10.21105.joss.01667.pdf 
#
#   [2] Paul A. Bereuter & Alois Sontacchi, "Influence of Different Input 
#       Features on Musical Source Separation Performance"; Fortschritte der
#       Akustik - DAGA 2023; p.430-433; 
#       URL: https://pub.dega-akustik.de/DAGA_2023/data/daga23_proceedings.pdf
#
#   [3] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Sylianos Ioannis Mimilakis & Rachel Bittner
#       "The MUSDB18 corpus for music separation"; Dec. 2017,
#       URL: https://sigsep.github.io/datasets/musdb.html#sisec-2018-evaluation-campaign
#
#
# Created by Paul A. Bereuter (February 2023)

import torch
import os
import toml
import copy
import string
import pandas
import tqdm
import numpy as np
import soundfile as sf
import pandas
from sklearn import preprocessing
from openunmix import data
from openunmix_slicq.transforms import ComplexNorm
from openunmix_slicq.CQT_class import VCQT
from openunmix.model_STFT import OpenUnmixComplex, make_filterbanks
from openunmix.model import OpenUnmix as OpenUnmixSpec
from openunmix.model import OpenUnmixLog
from openunmix.filtering import *
from utils.metrics import *


config_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(config_path)
dirs = toml.load(os.path.join(config_path, 'MSS_directories_local.toml'))
MUSDB_TEST_PATH = dirs['musdb_test']
# test data tuned for VQT
MUSDB_TEST_TUNED_PATH = dirs['musdb_tuned_test']

#set your output paths for audio examples and evaluation scores here
AUDIO_OUTPUT_PATH = './separated_audio_examples'
SCORE_OUTPUT_PATH = './evaluation_scores'
os.makedirs(AUDIO_OUTPUT_PATH,exist_ok=True)
os.makedirs(SCORE_OUTPUT_PATH,exist_ok=True)


config_magspec = toml.load(os.path.join(config_path, 'open_unmix_MagSpec_config.toml'))
config_compspec = toml.load(os.path.join(config_path, 'open_unmix_CompSpec_config.toml'))
config_cqtspec = toml.load(os.path.join(config_path, 'open_unmix_VQTSpec_config.toml'))

# Path to folder with model checkpoints
log_root = '../Trained_models'
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
TRAIN_SEQ_DUR_SPEC = config_magspec['seq_dur']

TARGET_STR_COMP = config_compspec['target']
N_TARGETS_COMP = config_compspec['n_targets']
NFFT_COMP = config_compspec['nfft']
NBINS_COMP = int(NFFT_SPEC/2)+1
NHOP_COMP = config_compspec['nhop']
HIDDEN_SIZE_COMP = config_compspec['hidden_size']
BANDWIDTH_COMP = config_compspec['bandwidth']
NB_CHANNELS_COMP = config_compspec['nb_channels']
N_LAYERS_COMP = config_compspec['nb_layers']
UNI_DIR_FLG_COMP = config_compspec['unidirectional']

F_MIN_CQT = config_cqtspec['fmin']
N_BINS_PER_OCT_CQT = config_cqtspec['bins_per_octave'] # e.g. 72 => divide one semitone into 6 bins (12th-tone)
QVAR_CQT = config_cqtspec['qvar'] # if set to one => constant-q-case
F_MAX_CQT = FS/2#f_min*2**n_octs
NBINS_CQT = int(np.ceil(np.log2(F_MAX_CQT/F_MIN_CQT)*N_BINS_PER_OCT_CQT))+1
BANDWIDTH_CQT = config_cqtspec['bandwidth']
HIDDEN_SIZE_CQT = config_cqtspec['hidden_size']
NB_CHANNELS_CQT = config_cqtspec['nb_channels']
N_LAYERS_CQT = config_cqtspec['nb_layers']
UNI_DIR_FLG_CQT = config_cqtspec['unidirectional']

vcqt_config={}
vcqt_config['fmin'] = F_MIN_CQT
vcqt_config['fmax'] = F_MAX_CQT
vcqt_config['nbins'] = N_BINS_PER_OCT_CQT
vcqt_config['qvar'] = QVAR_CQT
TRAIN_SEQ_DUR_CQT = config_cqtspec['seq_dur']

def get_subfoldernames_in_folder(folderPath: string):

    subfolder_list = [f.path for f in os.scandir(folderPath) if f.is_dir()]

    return subfolder_list

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


def get_statistics(encoder, dataset, verbose_Flag, log_flag):
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=verbose_Flag)
    for ind in pbar:
        mix, _, _, _, _, _  = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        if log_flag:
            X = torch.log(encoder(mix[None, ...]).mean(1, keepdim=False)).permute(0, 2, 1)
        else:
            X = encoder(mix[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))

    return scaler.mean_, std

def apply_noisy_phase(enh_spec:torch.Tensor, mix_stft: torch.Tensor, decoder):
    angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]

    enh_stft = torch.zeros(
        mix_stft.shape, dtype=mix_stft.dtype, device=mix_stft.device
    )
    enh_stft[..., 0] = (enh_spec.unsqueeze(-1).permute(0,3,2,1,4) * torch.cos(angle)).squeeze()
    enh_stft[..., 1] = (enh_spec.unsqueeze(-1).permute(0,3,2,1,4) * torch.sin(angle)).squeeze()
    enh_audio = decoder(enh_stft).permute(0,2,1)
    residual_stft = mix_stft-enh_stft
    residual_audio = decoder(residual_stft).permute(0,2,1)
    estimates_audio = torch.stack((enh_audio,residual_audio),0)

    return estimates_audio

def main():

    eval_device = torch.device("cpu")

    # initialize models and maximum bins for STFT and VQT
    temp_vcqt = VCQT(F_MIN_CQT, F_MAX_CQT, NBINS_CQT, Qvar=QVAR_CQT, fs=FS, audio_len=int(TRAIN_SEQ_DUR_CQT*FS), multichannel=True, device = eval_device, split_0_nyq=False)
    comp_norm = ComplexNorm(mono= NB_CHANNELS_SPEC == 1)
    MAX_BIN_SPEC = bandwidth_to_max_bin(FS, NFFT_SPEC, BANDWIDTH_SPEC)
    MAX_BIN_CQT = sum(temp_vcqt.cq.frqs<BANDWIDTH_CQT)-1 
    MagSpec_w_LR_sched_w_rand_mix_MSE = OpenUnmixSpec(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)
    MagSpec_w_LR_sched_wo_rand_mix_MSE = OpenUnmixSpec(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)
    MagSpec_wo_LR_sched_w_rand_mix_MSE = OpenUnmixSpec(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)    
    MagSpec_wo_LR_sched_w_rand_mix_CCMSE = OpenUnmixSpec(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)    
    CompSpec_wo_LR_sched_w_rand_mix_MSE = OpenUnmixComplex(nb_bins=NBINS_COMP, nb_channels=NB_CHANNELS_COMP, hidden_size=HIDDEN_SIZE_COMP, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_COMP, unidirectional=UNI_DIR_FLG_COMP, input_mean=None, input_scale=None)
    CompSpec_wo_LR_sched_w_rand_mix_CCMSE = OpenUnmixComplex(nb_bins=NBINS_COMP, nb_channels=NB_CHANNELS_COMP, hidden_size=HIDDEN_SIZE_COMP, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_COMP, unidirectional=UNI_DIR_FLG_COMP, input_mean=None, input_scale=None)
    CQTSpec_wo_LR_sched_w_rand_mix_MSE = OpenUnmixSpec(nb_bins=NBINS_CQT+1, nb_channels=NB_CHANNELS_CQT, hidden_size=HIDDEN_SIZE_CQT, max_bin = MAX_BIN_CQT, nb_layers=N_LAYERS_CQT, unidirectional=UNI_DIR_FLG_CQT, input_mean=None, input_scale=None)
    MagSpecLog_wo_LR_sched_w_rand_mix_MSE = OpenUnmixLog(nb_bins=NBINS_SPEC, nb_channels=NB_CHANNELS_SPEC, hidden_size=HIDDEN_SIZE_SPEC, max_bin = MAX_BIN_SPEC, nb_layers=N_LAYERS_SPEC, unidirectional=UNI_DIR_FLG_SPEC, input_mean=None, input_scale=None)

    #load checkpoints
    chosen_epoch_MagSpec_w_LR_sched_w_rand_mix_MSE = 999
    CKPT_PATH_MagSpec_w_LR_sched_w_rand_mix_MSE = log_root+'/01_DataGenComp_random_mixing/checkpoints/w_random_mixing/Open_Unmix_model_'+str(chosen_epoch_MagSpec_w_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_MagSpec_w_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_MagSpec_w_LR_sched_w_rand_mix_MSE, map_location=eval_device)
    MagSpec_w_LR_sched_w_rand_mix_MSE.load_state_dict(state_dict_MagSpec_w_LR_sched_w_rand_mix_MSE)
    MagSpec_w_LR_sched_w_rand_mix_MSE.eval()

    chosen_epoch_MagSpec_w_LR_sched_w_rand_mix_MSE = 959
    CKPT_PATH_MagSpec_w_LR_sched_wo_rand_mix_MSE = log_root+'/01_DataGenComp_random_mixing/checkpoints/no_random_mixing/Open_Unmix_model_'+str(chosen_epoch_MagSpec_w_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_MagSpec_w_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_MagSpec_w_LR_sched_wo_rand_mix_MSE, map_location=eval_device)
    MagSpec_w_LR_sched_wo_rand_mix_MSE.load_state_dict(state_dict_MagSpec_w_LR_sched_w_rand_mix_MSE)
    MagSpec_w_LR_sched_wo_rand_mix_MSE.eval()

    chosen_epoch_MagSpec_wo_LR_sched_w_rand_mix_CCMSE = 959
    CKPT_PATH_MagSpec_wo_LR_sched_w_rand_mix_CCMSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/MagSpec_CCMSE/Open_Unmix_model_'+str(chosen_epoch_MagSpec_wo_LR_sched_w_rand_mix_CCMSE)+'.ckpt'
    state_dict_MagSpec_wo_LR_sched_w_rand_mix_CCMSE = torch.load(CKPT_PATH_MagSpec_wo_LR_sched_w_rand_mix_CCMSE, map_location=eval_device)
    MagSpec_wo_LR_sched_w_rand_mix_CCMSE.load_state_dict(state_dict_MagSpec_wo_LR_sched_w_rand_mix_CCMSE)
    MagSpec_wo_LR_sched_w_rand_mix_CCMSE.eval()

    chosen_epoch_CompSpec_wo_LR_sched_w_rand_mix_MSE = 999
    CKPT_PATH_CompSpec_wo_LR_sched_w_rand_mix_MSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/CompSpec_MSE/Open_Unmix_model_'+str(chosen_epoch_CompSpec_wo_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_CompSpec_wo_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_CompSpec_wo_LR_sched_w_rand_mix_MSE, map_location=eval_device)
    CompSpec_wo_LR_sched_w_rand_mix_MSE.load_state_dict(state_dict_CompSpec_wo_LR_sched_w_rand_mix_MSE)
    CompSpec_wo_LR_sched_w_rand_mix_MSE.eval()

    chosen_epoch_CompSpec_wo_LR_sched_w_rand_mix_CCMSE = 849
    CKPT_PATH_CompSpec_wo_LR_sched_w_rand_mix_CCMSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/CompSpec_CCMSE/Open_Unmix_model_'+str(chosen_epoch_CompSpec_wo_LR_sched_w_rand_mix_CCMSE)+'.ckpt'
    state_dict_CompSpec_wo_LR_sched_w_rand_mix_CCMSE = torch.load(CKPT_PATH_CompSpec_wo_LR_sched_w_rand_mix_CCMSE, map_location=eval_device)
    CompSpec_wo_LR_sched_w_rand_mix_CCMSE.load_state_dict(state_dict_CompSpec_wo_LR_sched_w_rand_mix_CCMSE)
    CompSpec_wo_LR_sched_w_rand_mix_CCMSE.eval()

    chosen_epoch_CompSpec_wo_LR_sched_w_rand_mix_CCMSE = 849
    CKPT_PATH_CompSpec_wo_LR_sched_w_rand_mix_CCMSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/CompSpec_CCMSE/Open_Unmix_model_'+str(chosen_epoch_CompSpec_wo_LR_sched_w_rand_mix_CCMSE)+'.ckpt'
    state_dict_CompSpec_wo_LR_sched_w_rand_mix_CCMSE = torch.load(CKPT_PATH_CompSpec_wo_LR_sched_w_rand_mix_CCMSE, map_location=eval_device)
    CompSpec_wo_LR_sched_w_rand_mix_CCMSE.load_state_dict(state_dict_CompSpec_wo_LR_sched_w_rand_mix_CCMSE)
    CompSpec_wo_LR_sched_w_rand_mix_CCMSE.eval()

    chosen_epoch_CQTSpec_wo_LR_sched_w_rand_mix_MSE = 959
    CKPT_PATH_CQTSpec_wo_LR_sched_w_rand_mix_MSE =log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/CQTSpec_MSE/Open_Unmix_model_'+str(chosen_epoch_CQTSpec_wo_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_CQTSpec_wo_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_CQTSpec_wo_LR_sched_w_rand_mix_MSE, map_location=eval_device)
    CQTSpec_wo_LR_sched_w_rand_mix_MSE.load_state_dict(state_dict_CQTSpec_wo_LR_sched_w_rand_mix_MSE)
    CQTSpec_wo_LR_sched_w_rand_mix_MSE.eval()

    chosen_epoch_MagSpecLog_wo_LR_sched_w_rand_mix_MSE = 879
    CKPT_PATH_MagSpecLog_wo_LR_sched_w_rand_mix_MSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/MagSpecLog_MSE_w_LR_schedule/Open_Unmix_model_'+str(chosen_epoch_MagSpecLog_wo_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_MagSpecLog_wo_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_MagSpecLog_wo_LR_sched_w_rand_mix_MSE, map_location=eval_device)
    MagSpecLog_wo_LR_sched_w_rand_mix_MSE.load_state_dict(state_dict_MagSpecLog_wo_LR_sched_w_rand_mix_MSE)
    MagSpecLog_wo_LR_sched_w_rand_mix_MSE.eval()

    chosen_epoch_MagSpec_wo_LR_sched_w_rand_mix_MSE = 799
    CKPT_PATH_MagSpec_wo_LR_sched_w_rand_mix_MSE = log_root+'/03_LossComp_MSE_vs_CCMSE/checkpoints/MagSpec_MSE_wo_LR_schedule/Open_Unmix_model_'+str(chosen_epoch_MagSpec_wo_LR_sched_w_rand_mix_MSE)+'.ckpt'
    state_dict_MagSpec_wo_LR_sched_w_rand_mix_MSE = torch.load(CKPT_PATH_MagSpec_wo_LR_sched_w_rand_mix_MSE, map_location=eval_device)
    MagSpec_wo_LR_sched_w_rand_mix_MSE.load_state_dict(state_dict_MagSpec_wo_LR_sched_w_rand_mix_MSE)
    MagSpec_wo_LR_sched_w_rand_mix_MSE.eval()

    #load data
    musdb_test_folderlist = get_subfoldernames_in_folder(MUSDB_TEST_PATH)
    musdb_test_tuned_folderlist = get_subfoldernames_in_folder(MUSDB_TEST_TUNED_PATH)
    N_files = len(musdb_test_folderlist)
    
    # preallocate score arrays
    test_scores_model_1 = torch.zeros((4,2,N_files))
    test_scores_model_2 = torch.zeros((4,2,N_files))
    test_scores_model_3 = torch.zeros((4,2,N_files))
    test_scores_model_4 = torch.zeros((4,2,N_files))
    test_scores_model_5 = torch.zeros((4,2,N_files))
    test_scores_model_6 = torch.zeros((4,2,N_files))
    test_scores_model_7 = torch.zeros((4,2,N_files))
    test_scores_model_8 = torch.zeros((4,2,N_files))


    files_to_save = ['Al James - Schoolboy Facination','The Long Wait - Dark Horses','Sambasevam Shanmugam - Kaathaadi','Speak Softly - Broken Man','Timboz - Pony','Hollow Ground - Ill Fate','The Sunshine Garcia Band - For I Am The Moon']
    # iterate over data separate vocals from mixtures and evaluate metrics
    for ii in (pbar:=(tqdm.tqdm(range(N_files)))):
        # prepare audio paths
        pbar.set_description('Processing file #'+str(ii))
        song_name = musdb_test_folderlist[ii].split('/')[-1]
        save_Flag = True if song_name in files_to_save else False
        mix_filepath = os.path.join(musdb_test_folderlist[ii],'mixture.wav')
        mix_filepath_tuned = os.path.join(musdb_test_tuned_folderlist[ii],'mixture.wav')
        target_filepath = os.path.join(musdb_test_folderlist[ii],'vocals.wav')
        target_filepath_tuned = os.path.join(musdb_test_tuned_folderlist[ii],'vocals.wav')

        # load mixture
        mix, _ = sf.read(mix_filepath)
        mix = torch.Tensor(mix.T).unsqueeze(0)
        mix_tuned, fs = sf.read(mix_filepath_tuned)
        mix_tuned = torch.Tensor(mix_tuned.T).unsqueeze(0)
        audio_len = mix.size(-1)
        audio_len_tuned = mix_tuned.size(-1)

        # load target audio
        target, _ = sf.read(target_filepath)
        target = torch.Tensor(target.T).unsqueeze(0)
        target_tuned, _ = sf.read(target_filepath_tuned)
        target_tuned = torch.Tensor(target_tuned.T).unsqueeze(0)

        # calculate groundtruth residual/accompaniment 
        residual = mix - target
        residual_tuned = mix_tuned - target_tuned

        stft, istft = make_filterbanks(n_fft=NFFT_SPEC, n_hop=NHOP_SPEC, sample_rate=FS, center=True, length = audio_len)
        mix_stft = stft(mix)
        mix_spec = comp_norm(mix_stft)[0].unsqueeze(0)

        vcqt = VCQT(F_MIN_CQT, F_MAX_CQT, NBINS_CQT, Qvar=QVAR_CQT, fs=FS, audio_len=audio_len_tuned, multichannel=True, device = eval_device, split_0_nyq=False)
        ivcqt = lambda x: vcqt.bwd(x)

        mix_vcqt = vcqt.fwd(mix_tuned)
        mix_vcqt_spec = comp_norm(mix_vcqt)[0].unsqueeze(0)
        if save_Flag:
            os.makedirs(os.path.join(AUDIO_OUTPUT_PATH,song_name),exist_ok=True)


        #separate vocals and residual => "enhancement"
        enh_spec_model_1 = MagSpec_w_LR_sched_w_rand_mix_MSE(mix_spec)
        est_audio_model_1 = apply_noisy_phase(enh_spec_model_1, mix_stft,istft).detach()
        if save_Flag:
            os.makedirs(os.path.join(AUDIO_OUTPUT_PATH,song_name),exist_ok=True)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_1.wav'),est_audio_model_1[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_1.wav'),est_audio_model_1[1,...].squeeze(), samplerate = FS)

        enh_spec_model_2 = MagSpec_w_LR_sched_wo_rand_mix_MSE(mix_spec)
        est_audio_model_2 = apply_noisy_phase(enh_spec_model_2, mix_stft,istft).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_2.wav'),est_audio_model_2[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_2.wav'),est_audio_model_2[1,...].squeeze(), samplerate = FS)

        enh_spec_model_3 = MagSpec_wo_LR_sched_w_rand_mix_CCMSE(mix_spec)
        est_audio_model_3 = apply_noisy_phase(enh_spec_model_3, mix_stft,istft).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_3.wav'),est_audio_model_3[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_3.wav'),est_audio_model_3[1,...].squeeze(), samplerate = FS)

        enh_spec_model_4 = CompSpec_wo_LR_sched_w_rand_mix_MSE(mix_stft)
        enh_audio_model_4 = istft(enh_spec_model_4)
        est_audio_model_4 = torch.stack((enh_audio_model_4,mix-enh_audio_model_4),0).permute(0,1,3,2).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_4.wav'),est_audio_model_4[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_4.wav'),est_audio_model_4[1,...].squeeze(), samplerate = FS)

        enh_spec_model_5 = CompSpec_wo_LR_sched_w_rand_mix_CCMSE(mix_stft)
        enh_audio_model_5 = istft(enh_spec_model_5)
        est_audio_model_5 = torch.stack((enh_audio_model_5,mix-enh_audio_model_5),0).permute(0,1,3,2).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_5.wav'),est_audio_model_5[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_5.wav'),est_audio_model_5[1,...].squeeze(), samplerate = FS)

        enh_spec_model_6 = CQTSpec_wo_LR_sched_w_rand_mix_MSE(mix_vcqt_spec)
        est_audio_model_6 = apply_noisy_phase(enh_spec_model_6, mix_vcqt, ivcqt).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_6.wav'),est_audio_model_6[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_6.wav'),est_audio_model_6[1,...].squeeze(), samplerate = FS)

        enh_spec_model_7 = MagSpecLog_wo_LR_sched_w_rand_mix_MSE(mix_spec)
        est_audio_model_7 = apply_noisy_phase(enh_spec_model_7, mix_stft,istft).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_7.wav'),est_audio_model_7[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_7.wav'),est_audio_model_7[1,...].squeeze(), samplerate = FS)

        enh_spec_model_8 = MagSpec_wo_LR_sched_w_rand_mix_MSE(mix_spec)
        est_audio_model_8 = apply_noisy_phase(enh_spec_model_8, mix_stft,istft).detach()
        if save_Flag:
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_model_8.wav'),est_audio_model_8[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_model_8.wav'),est_audio_model_8[1,...].squeeze(), samplerate = FS)

        # #calculate metrics
        targets = torch.stack((target.permute(0,2,1), residual.permute(0,2,1)), dim=0)
        targets_tuned = torch.stack((target_tuned.permute(0,2,1), residual_tuned.permute(0,2,1)), dim=0)

        scores_model_1 = np.array(calc_metrics(targets.squeeze(), est_audio_model_1.squeeze(), False))
        test_scores_model_1[...,ii] = torch.Tensor(scores_model_1)

        scores_model_2 = np.array(calc_metrics(targets.squeeze(), est_audio_model_2.squeeze(), False))
        test_scores_model_2[...,ii] = torch.Tensor(scores_model_2)

        scores_model_3 = np.array(calc_metrics(targets.squeeze(), est_audio_model_3.squeeze(), False))
        test_scores_model_3[...,ii] = torch.Tensor(scores_model_3)

        scores_model_4 = np.array(calc_metrics(targets.squeeze(), est_audio_model_4.squeeze(), False))
        test_scores_model_4[...,ii] = torch.Tensor(scores_model_4)

        scores_model_5 = np.array(calc_metrics(targets.squeeze(), est_audio_model_5.squeeze(), False))
        test_scores_model_5[...,ii] = torch.Tensor(scores_model_5)

        scores_model_6 = np.array(calc_metrics(targets_tuned.squeeze(), est_audio_model_6.squeeze(), False))
        test_scores_model_6[...,ii] = torch.Tensor(scores_model_6)
        
        scores_model_7 = np.array(calc_metrics(targets.squeeze(), est_audio_model_7.squeeze(), False))
        test_scores_model_7[...,ii] = torch.Tensor(scores_model_7)

        scores_model_8 = np.array(calc_metrics(targets.squeeze(), est_audio_model_8.squeeze(), False))
        test_scores_model_8[...,ii] = torch.Tensor(scores_model_8)

    #save metrics
    test_scores_vocals_model_1_dict = {'sdr':test_scores_model_1[0,0,:],'isr':test_scores_model_1[1,0,:],'sir':test_scores_model_1[2,0,:],'sar':test_scores_model_1[3,0,:]}
    test_scores_residual_model_1_dict = {'sdr':test_scores_model_1[0,1,:],'isr':test_scores_model_1[1,1,:],'sir':test_scores_model_1[2,1,:],'sar':test_scores_model_1[3,1,:]}
    df_vocals_model_1 = pandas.DataFrame(test_scores_vocals_model_1_dict)
    df_residual_model_1 = pandas.DataFrame(test_scores_residual_model_1_dict)
    df_vocals_model_1.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_1.csv'))
    df_residual_model_1.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_1.csv'))

    test_scores_vocals_model_2_dict = {'sdr':test_scores_model_2[0,0,:],'isr':test_scores_model_2[1,0,:],'sir':test_scores_model_2[2,0,:],'sar':test_scores_model_2[3,0,:]}
    test_scores_residual_model_2_dict = {'sdr':test_scores_model_2[0,1,:],'isr':test_scores_model_2[1,1,:],'sir':test_scores_model_2[2,1,:],'sar':test_scores_model_2[3,1,:]}
    df_vocals_model_2 = pandas.DataFrame(test_scores_vocals_model_2_dict)
    df_residual_model_2 = pandas.DataFrame(test_scores_residual_model_2_dict)
    df_vocals_model_2.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_2.csv'))
    df_residual_model_2.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_2.csv'))

    test_scores_vocals_model_3_dict = {'sdr':test_scores_model_3[0,0,:],'isr':test_scores_model_3[1,0,:],'sir':test_scores_model_3[2,0,:],'sar':test_scores_model_3[3,0,:]}
    test_scores_residual_model_3_dict = {'sdr':test_scores_model_3[0,1,:],'isr':test_scores_model_3[1,1,:],'sir':test_scores_model_3[2,1,:],'sar':test_scores_model_3[3,1,:]}
    df_vocals_model_3 = pandas.DataFrame(test_scores_vocals_model_3_dict)
    df_residual_model_3 = pandas.DataFrame(test_scores_residual_model_3_dict)
    df_vocals_model_3.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_3.csv'))
    df_residual_model_3.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_3.csv'))

    test_scores_vocals_model_4_dict = {'sdr':test_scores_model_4[0,0,:],'isr':test_scores_model_4[1,0,:],'sir':test_scores_model_4[2,0,:],'sar':test_scores_model_4[3,0,:]}
    test_scores_residual_model_4_dict = {'sdr':test_scores_model_4[0,1,:],'isr':test_scores_model_4[1,1,:],'sir':test_scores_model_4[2,1,:],'sar':test_scores_model_4[3,1,:]}
    df_vocals_model_4 = pandas.DataFrame(test_scores_vocals_model_4_dict)
    df_residual_model_4 = pandas.DataFrame(test_scores_residual_model_4_dict)
    df_vocals_model_4.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_4.csv'))
    df_residual_model_4.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_4.csv'))

    test_scores_vocals_model_5_dict = {'sdr':test_scores_model_5[0,0,:],'isr':test_scores_model_5[1,0,:],'sir':test_scores_model_5[2,0,:],'sar':test_scores_model_5[3,0,:]}
    test_scores_residual_model_5_dict = {'sdr':test_scores_model_5[0,1,:],'isr':test_scores_model_5[1,1,:],'sir':test_scores_model_5[2,1,:],'sar':test_scores_model_5[3,1,:]}
    df_vocals_model_5 = pandas.DataFrame(test_scores_vocals_model_5_dict)
    df_residual_model_5 = pandas.DataFrame(test_scores_residual_model_5_dict)
    df_vocals_model_5.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_5.csv'))
    df_residual_model_5.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_5.csv'))

    test_scores_vocals_model_6_dict = {'sdr':test_scores_model_6[0,0,:],'isr':test_scores_model_6[1,0,:],'sir':test_scores_model_6[2,0,:],'sar':test_scores_model_6[3,0,:]}
    test_scores_residual_model_6_dict = {'sdr':test_scores_model_6[0,1,:],'isr':test_scores_model_6[1,1,:],'sir':test_scores_model_6[2,1,:],'sar':test_scores_model_6[3,1,:]}
    df_vocals_model_6 = pandas.DataFrame(test_scores_vocals_model_6_dict)
    df_residual_model_6 = pandas.DataFrame(test_scores_residual_model_6_dict)
    df_vocals_model_6.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_6.csv'))
    df_residual_model_6.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_6.csv'))

    test_scores_vocals_model_7_dict = {'sdr':test_scores_model_7[0,0,:],'isr':test_scores_model_7[1,0,:],'sir':test_scores_model_7[2,0,:],'sar':test_scores_model_7[3,0,:]}
    test_scores_residual_model_7_dict = {'sdr':test_scores_model_7[0,1,:],'isr':test_scores_model_7[1,1,:],'sir':test_scores_model_7[2,1,:],'sar':test_scores_model_7[3,1,:]}
    df_vocals_model_7 = pandas.DataFrame(test_scores_vocals_model_7_dict)
    df_residual_model_7 = pandas.DataFrame(test_scores_residual_model_7_dict)
    df_vocals_model_7.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_7.csv'))
    df_residual_model_7.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_7.csv'))

    test_scores_vocals_model_8_dict = {'sdr':test_scores_model_8[0,0,:],'isr':test_scores_model_8[1,0,:],'sir':test_scores_model_8[2,0,:],'sar':test_scores_model_8[3,0,:]}
    test_scores_residual_model_8_dict = {'sdr':test_scores_model_8[0,1,:],'isr':test_scores_model_8[1,1,:],'sir':test_scores_model_8[2,1,:],'sar':test_scores_model_8[3,1,:]}
    df_vocals_model_8 = pandas.DataFrame(test_scores_vocals_model_8_dict)
    df_residual_model_8 = pandas.DataFrame(test_scores_residual_model_8_dict)
    df_vocals_model_8.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_vocals_model_8.csv'))
    df_residual_model_8.to_csv(os.path.join(SCORE_OUTPUT_PATH,'test_score_residual_model_8.csv'))

    print("This is the end. My only friend, the end. -Jim Morrison, John Paul Densmore, Robert Krieger, Raymond Manzarek (The Doors), 1967")

if __name__ == "__main__":
    main()
