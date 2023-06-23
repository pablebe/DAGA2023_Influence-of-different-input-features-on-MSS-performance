#%% Python script to evaluate noisy and IRM (stft and vqt-based) metrics for MUSDB18 dataset [1].
#   Results are used in [2] to compare performance of modified Open-Unmix framework [3].
#
#
#   References:
#   [1] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Sylianos Ioannis Mimilakis & Rachel Bittner
#       "The MUSDB18 corpus for music separation"; Dec. 2017,
#       URL: https://sigsep.github.io/datasets/musdb.html#sisec-2018-evaluation-campaign
#
#   [2] Paul A. Bereuter & Alois Sontacchi, "Influence of Different Input 
#       Features on Musical Source Separation Performance"; Fortschritte der
#       Akustik - DAGA 2023; p.430-433; 
#       URL: https://pub.dega-akustik.de/DAGA_2023/data/daga23_proceedings.pdf
#
#   [3] Fabian-Robert Stöter, Stefan Uhlich, Antoine Liutkus & Yuki Mitsufuji
#       "Open-Unmix - A Reference Implementation for Music"; Sept. 2019;
#       URL: https://www.theoj.org/joss-papers/joss.01667/10.21105.joss.01667.pdf 
#
#
# Created by Paul A. Bereuter (February 2023)

import torch
import os
import toml
import string
import pandas
import tqdm
import numpy as np
import soundfile as sf
import pandas
from openunmix_slicq.transforms import ComplexNorm
from openunmix_slicq.CQT_class import VCQT
from openunmix.model_STFT import make_filterbanks
from openunmix.filtering import *
from utils.metrics import *

config_path = os.path.dirname(os.path.realpath(__file__))
dirs = toml.load(os.path.join(config_path, 'MSS_directories_local.toml'))
MUSDB_TEST_PATH = dirs['musdb_test']
MUSDB_TEST_TUNED_PATH = dirs['musdb_tuned_test']
AUDIO_OUTPUT_PATH = '/Users/Paul/IEM-Phd/03_PhD/03_Experiments/Evaluation_open_unmix_models/separated_audio_examples'
SCORE_OUTPUT_PATH = '/Users/Paul/IEM-Phd/03_PhD/06_Conferences/daga2023/code/03_Experiments/Evaluation_open_unmix_models/evaluation_scores'

config_magspec = toml.load(os.path.join(config_path, 'open_unmix_MagSpec_config.toml'))
config_compspec = toml.load(os.path.join(config_path, 'open_unmix_CompSpec_config.toml'))
config_cqtspec = toml.load(os.path.join(config_path, 'open_unmix_CQTSpec_config.toml'))

FS = 44100

NB_CHANNELS_SPEC = config_magspec['nb_channels']

NFFT_SPEC = config_magspec['nfft']
NBINS_SPEC = int(NFFT_SPEC/2)+1
NHOP_SPEC = config_magspec['nhop']

def get_subfoldernames_in_folder(folderPath: string):

    subfolder_list = [f.path for f in os.scandir(folderPath) if f.is_dir()]

    return subfolder_list

def apply_noisy_phase(enh_spec:torch.Tensor, mix_stft: torch.Tensor, decoder):
    angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]

    enh_stft = torch.zeros(
        mix_stft.shape, dtype=mix_stft.dtype, device=mix_stft.device
    )
    enh_stft[..., 0] = (enh_spec.unsqueeze(-1) * torch.cos(angle)).squeeze()
    enh_stft[..., 1] = (enh_spec.unsqueeze(-1) * torch.sin(angle)).squeeze()
    enh_audio = decoder(enh_stft).permute(0,2,1)
    residual_stft = mix_stft-enh_stft
    residual_audio = decoder(residual_stft).permute(0,2,1)
    estimates_audio = torch.stack((enh_audio,residual_audio),0)

    return estimates_audio


F_MIN_CQT = config_cqtspec['fmin']
N_BINS_PER_OCT_CQT = config_cqtspec['bins_per_octave'] # e.g. 72 => divide one semitone into 6 bins (12th-tone)
QVAR_CQT = config_cqtspec['qvar'] # if set to one => constant-q-case
F_MAX_CQT = FS/2#f_min*2**n_octs
NBINS_CQT = int(np.ceil(np.log2(F_MAX_CQT/F_MIN_CQT)*N_BINS_PER_OCT_CQT))+1

def main():

    eval_device = torch.device("cpu")
    #load data
    musdb_test_folderlist = get_subfoldernames_in_folder(MUSDB_TEST_PATH)
    musdb_test_tuned_folderlist = get_subfoldernames_in_folder(MUSDB_TEST_TUNED_PATH)
    N_files = len(musdb_test_folderlist)
    noisy_scores_out = torch.zeros((4,1,N_files))
    irm_scores_out = torch.zeros((4,2,N_files))
    irmvq_scores_out = torch.zeros((4,2,N_files))

    comp_norm = ComplexNorm(mono= NB_CHANNELS_SPEC == 1)
    files_to_save = ['Al James - Schoolboy Facination','The Long Wait - Dark Horses','Sambasevam Shanmugam - Kaathaadi','Speak Softly - Broken Man','Timboz - Pony','Hollow Ground - Ill Fate','The Sunshine Garcia Band - For I Am The Moon']
    # iterate over data separate vocals from mixtures and evaluate metrics
    for ii in (pbar:=(tqdm.tqdm(range(N_files)))):
        pbar.set_description('Processing file #'+str(ii))

        song_name = musdb_test_folderlist[ii].split('/')[-1]
        save_Flag = True if song_name in files_to_save else False
        mix_filepath = os.path.join(musdb_test_folderlist[ii],'mixture.wav')
        mix_filepath_tuned = os.path.join(musdb_test_tuned_folderlist[ii],'mixture.wav')
        target_filepath = os.path.join(musdb_test_folderlist[ii],'vocals.wav')
        target_filepath_tuned = os.path.join(musdb_test_tuned_folderlist[ii],'vocals.wav')

        mix, _ = sf.read(mix_filepath)
        mix = torch.Tensor(mix.T).unsqueeze(0)
        target, _ = sf.read(target_filepath)
        target = torch.Tensor(target.T).unsqueeze(0)

        mix_tuned, _ = sf.read(mix_filepath_tuned)
        mix_tuned = torch.Tensor(mix_tuned.T).unsqueeze(0)
        target_tuned, _ = sf.read(target_filepath_tuned)
        target_tuned = torch.Tensor(target_tuned.T).unsqueeze(0)

        audio_len = mix.size(-1)
        audio_len_tuned = mix_tuned.size(-1)
        vqt = VCQT(F_MIN_CQT, F_MAX_CQT, NBINS_CQT, Qvar=QVAR_CQT, fs=FS, audio_len=int(audio_len_tuned), multichannel=True, device = "cpu", split_0_nyq=False)
        vqt_fwd = lambda x: vqt.fwd(x)
        vqt_bwd = lambda x: vqt.bwd(x)

        stft, istft = make_filterbanks(n_fft=NFFT_SPEC, n_hop=NHOP_SPEC, sample_rate=FS, center=True, length = audio_len)
        mix_stft = stft(mix)
        mix_spec = comp_norm(mix_stft)[0].unsqueeze(0)
        target_stft = stft(target)
        target_spec = comp_norm(target_stft)[0].unsqueeze(0)
        noise_stft = mix_stft-target_stft
        noise_audio = istft(noise_stft)
        noise_spec = comp_norm(noise_stft)[0].unsqueeze(0)#mix_spec-target_spec
        os.makedirs(os.path.join(AUDIO_OUTPUT_PATH,song_name),exist_ok=True)
        sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'noise_IRM.wav'),noise_audio.squeeze().t(), samplerate = FS)
        IRM = torch.sqrt((target_spec**2)/(target_spec**2+noise_spec**2+1e-6))

        target_vq = vqt_fwd(target_tuned)
        target_spec_vq = comp_norm(target_vq)[0].unsqueeze(0)
        mix_vq = vqt_fwd(mix_tuned)
        mix_spec_vq = comp_norm(mix_vq)[0].unsqueeze(0)
        noise_vq = mix_vq-target_vq
        noise_spec_vq = comp_norm(noise_vq)[0].unsqueeze(0)
        IRM_VQ = torch.sqrt((target_spec_vq**2)/(target_spec_vq**2+noise_spec_vq**2+1e-6))
        enh_spec_irm = (mix_spec*IRM)
        enh_spec_vq_irm = (mix_spec_vq*IRM_VQ)
        enh_audio_irm = apply_noisy_phase(enh_spec_irm, mix_stft, istft)#.detach()
        enh_audio_irmvq = apply_noisy_phase(enh_spec_vq_irm, mix_vq, vqt_bwd)

        if save_Flag:
            os.makedirs(os.path.join(AUDIO_OUTPUT_PATH,song_name),exist_ok=True)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_IRM.wav'),enh_audio_irm[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_IRM.wav'), enh_audio_irm[1,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_vocals_IRM_VQ.wav'), enh_audio_irmvq[0,...].squeeze(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'separated_residual_IRM_VQ.wav'),enh_audio_irmvq[1,...].squeeze(), samplerate = FS)


        if save_Flag:
            os.makedirs(os.path.join(AUDIO_OUTPUT_PATH,song_name),exist_ok=True)

        if save_Flag:
            os.makedirs(os.path.join(AUDIO_OUTPUT_PATH,song_name),exist_ok=True)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'mix.wav'),mix.squeeze().t(), samplerate = FS)
            sf.write(os.path.join(AUDIO_OUTPUT_PATH,song_name,'target.wav'),target.squeeze().t(), samplerate = FS)
        
        #calculate metrics
        #for noisy metrics use mix as if it was the enhanced file!
        residual = mix-target 
        targets = torch.stack((target.permute(0,2,1), residual.permute(0,2,1)), dim=0)#stack such that number of sources is at dimension 0
        irm_scores = np.array(calc_metrics(targets.squeeze(), enh_audio_irm.squeeze(), False))
        irm_scores_out[...,ii] = torch.Tensor(irm_scores)

        residual_tuned = mix_tuned-target_tuned
        targets_tuned = torch.stack((target_tuned.permute(0,2,1), residual_tuned.permute(0,2,1)), dim=0)#stack such that number of sources is at dimension 0
        irmvq_scores = np.array(calc_metrics(targets_tuned.squeeze(), enh_audio_irmvq.squeeze(), False))
        irmvq_scores_out[...,ii] = torch.Tensor(irmvq_scores)

        noisy_scores = np.array(calc_metrics(target.squeeze().t().unsqueeze(0), mix.squeeze().t().unsqueeze(0), False))
        noisy_scores_out[...,ii] = torch.Tensor(noisy_scores)

    #save metrics
    noisy_scores = {'sdr':noisy_scores_out[0,0,:],'isr':noisy_scores_out[1,0,:],'sir':noisy_scores_out[2,0,:],'sar':noisy_scores_out[3,0,:]}
    noisy_scores_pd = pandas.DataFrame(noisy_scores)
    noisy_scores_pd.to_csv(os.path.join(SCORE_OUTPUT_PATH,'noisy_scores.csv'))

    irm_scores = {'sdr':irm_scores_out[0,0,:],'isr':irm_scores_out[1,0,:],'sir':irm_scores_out[2,0,:],'sar':irm_scores_out[3,0,:]}
    irm_scores_pd = pandas.DataFrame(irm_scores)
    irm_scores_pd.to_csv(os.path.join(SCORE_OUTPUT_PATH,'IRM_scores_vocals.csv'))
    
    irm_scores = {'sdr':irm_scores_out[0,0,:],'isr':irm_scores_out[1,1,:],'sir':irm_scores_out[2,1,:],'sar':irm_scores_out[3,1,:]}
    irm_scores_pd = pandas.DataFrame(irm_scores)
    irm_scores_pd.to_csv(os.path.join(SCORE_OUTPUT_PATH,'IRM_scores_residual.csv'))

    irmvq_scores = {'sdr':irmvq_scores_out[0,0,:],'isr':irmvq_scores_out[1,0,:],'sir':irmvq_scores_out[2,0,:],'sar':irmvq_scores_out[3,0,:]}
    irmvq_scores_pd = pandas.DataFrame(irmvq_scores)
    irmvq_scores_pd.to_csv(os.path.join(SCORE_OUTPUT_PATH,'IRMVQ_scores_vocals.csv'))
    
    irmvq_scores = {'sdr':irmvq_scores_out[0,0,:],'isr':irmvq_scores_out[1,1,:],'sir':irmvq_scores_out[2,1,:],'sar':irmvq_scores_out[3,1,:]}
    irmvq_scores_pd = pandas.DataFrame(irmvq_scores)
    irmvq_scores_pd.to_csv(os.path.join(SCORE_OUTPUT_PATH,'IRMVQ_scores_residual.csv'))

    print("This is the end. My only friend, the end. -Jim Morrison, John Paul Densmore, Robert Krieger, Raymond Manzarek (The Doors), 1967")


if __name__ == "__main__":
    main()
