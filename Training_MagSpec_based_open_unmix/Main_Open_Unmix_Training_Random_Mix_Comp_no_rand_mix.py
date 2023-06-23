#%% Python script to train Open-Unmix model [1] without random mixing. 
#   Allows the training of the MagSpec models from [2], which are all Open-Unmix models 
#   with certain modifications within the UMX-framework.
#   The results of the model are compared to other modified Open-Unmix models in [2].
#   The models are trained using the MUSDB Dataset from [3].
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
# Created by Paul A. Bereuter (February 2023)

import torch
import torchaudio
import toml
import platform
import gc
import tqdm
import copy
import os
import string
import numpy as np
import ptflops
import random
import sys
from torch.utils.tensorboard import SummaryWriter
from openunmix import data
from sklearn import preprocessing
from openunmix.model import OpenUnmix, OpenUnmixLog, make_filterbanks, ComplexNorm
from openunmix.filtering import * 
from torch import Tensor
from utils.MSS_dataset import MSSMUSDBDataset
from torch.utils.data import DataLoader
from utils.metrics import *
from typing import Union

config_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(config_path)

os.environ['CUDA_VISIBLE_DEVICES']='1'
RANDOM_MIX_FLAG = True  # if True => random mixing training strategy
USE_LR_SCHEDULER = False # if True => training with LR-schedule
USE_LOG_FEATURES = False # if True => MagSpecLog model (log-features) are used for training

if platform.system() == 'Linux':
    def numpy_random_seed(ind=None):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    worker_init_fn_rand = numpy_random_seed
else:
    worker_init_fn_rand = None


config = toml.load(os.path.join(config_path, 'open_unmix_MagSpec_config.toml'))
dirs = toml.load(os.path.join(config_path, 'MSS_directories_remote.toml'))
Tensorboard_PATH = config['log_path']
MUSDB_PATH = dirs['musdb_root']
CKPT_PATH = config['checkpoint_path']

FS = 44100
TARGET_STR = config['target']
N_TARGETS = config['n_targets']
TRAIN_FILEPATH = dirs['musdb_train']
LOG_PATH = config['log_path']
NFFT = config['nfft']
NBINS = int(NFFT/2)+1
NHOP = config['nhop']
HIDDEN_SIZE = config['hidden_size']
BANDWIDTH = config['bandwidth']
NB_CHANNELS = config['nb_channels']
N_LAYERS = config['nb_layers']
UNI_DIR_FLG = config['unidirectional']
VERBOSE = config['quiet']
USE_WIENER = config['use_wiener']
WIENER_LEN = config['wiener_len']
N_WIENER_ITs = config['n_wiener_its']
WIENER_SOFTMAX_INIT = config['wiener_softmax_init']
COMPUTE_RESIDUAL = config['compute_residual']

train_config = config['train_config']
START_w_VALID = train_config['start_w_valid']
TRAIN_DATA_PERCENTAGE = train_config['train_data_percentage']
SAMPLES_PER_TRACK_TRAIN = train_config['samples_per_track']
BATCH_SIZE_TRAIN = train_config['train_batch_size']
NUM_PROCESSES_TRAIN_LOADER = train_config['num_train_workers']
LR = train_config['lr']
WEIGHT_DECAY = train_config['weight_decay']
MAX_EPOCHS = train_config['epochs']
SEQ_DUR_TRAIN = train_config['seq_dur']
PATIENCE = train_config['patience']
LR_DECAY_PATIENCE = train_config['lr_decay_patience']
LR_DECAY_GAMMA = train_config['lr_decay_gamma']

valid_config = config['valid_config']
BATCH_SIZE_VALID = valid_config['valid_batch_size']
SAMPLES_PER_TRACK_VALID = valid_config['samples_per_track']
NUM_PROCESSES_VALID_LOADER = valid_config['num_train_workers']
SEQ_DUR_VALID = valid_config['valid_seq_dur']
N_LOG_EPOCHS = 1 #number of epochs after which validation is run and checkpoints are stored


if RANDOM_MIX_FLAG:
    log_writer = SummaryWriter(os.path.join(Tensorboard_PATH,'w_random_mixing'),filename_suffix='.tlog')
    CKPT_PATH = os.path.join(CKPT_PATH,'w_random_mixing')

else:
    log_writer = SummaryWriter(os.path.join(Tensorboard_PATH,'no_random_mixing'),filename_suffix='.tlog')
    CKPT_PATH = os.path.join(CKPT_PATH,'no_random_mixing')

def numpy_fixed_seed(ind=None):
    np.random.seed(valid_config['fixed_seed']+ind)

def get_subfoldernames_in_folder(folderPath: string):

    subfolder_list = [f.path for f in os.scandir(folderPath) if f.is_dir()]

    return subfolder_list


def get_model_summary(FS: int, model, device):
    # helper function to print model summary using the ptflops package
    # prints Model summary and complexity metrics (MACs/s and #(params))

    x = torch.randn([1,NB_CHANNELS,NBINS,int((FS)/NHOP)]).to(device)

    macs_overall, params_overall = ptflops.get_model_complexity_info(
        model,
        (2,FS),
        input_constructor = lambda _: {"x": x},
        as_strings = False,
        verbose = True,
    )

    print('{:<30}  {:<8}'.format('Computational complexity (incl. STFT, Feature-Norm and ISTFT): ', str(macs_overall/1e6)),' M MACs/s')
    print('{:<30}  {:<8}'.format('Number of parameters (incl. STFT, Feature-Norm and ISTFT): ', str(params_overall/1e6)),' M parameters')


def write_checkpoint(obj: Union[torch.optim.Optimizer, torch.nn.Module], name: str, dir: str, epoch: int, extension="ckpt",):
    filename = name+ str(epoch)+"."+extension
    cp_name = os.path.join(dir, filename)
    torch.save(obj.state_dict(), cp_name)
    print("Checkpoint '"+filename+"' for epoch "+str(epoch)+" has been stored.")

def load_checkpoint_no_scheduler(dirname: str, file_list: str, extension: str):
    # get latest checkpoint
    epochs = [i.split('_', -1)[-1] for i in file_list]
    epochs = [int(i.split('.', -1)[0]) for i in epochs]
    latest_epoch = max(epochs)    
    latest_substring = '_'+str(latest_epoch)+extension
    latest_ckpts = [latest_substring in d for d in file_list]
    temp = np.array(file_list)
    latest_ckpt_files = temp[latest_ckpts]

    try:
        assert len(latest_ckpt_files)==2
    except AssertionError:
        sys.exit("There exist either too many checkpoint-files or one checkpoint-file is missing!")

    model_idx = np.array(["model" in f for f in latest_ckpt_files])
    latest_model_ckpt = latest_ckpt_files[model_idx][0]
    opt_idx = np.array(["optimizer" in f for f in latest_ckpt_files])
    latest_opt_ckpt = latest_ckpt_files[opt_idx][0]

    model_state_dict = torch.load(os.path.join(dirname,latest_model_ckpt),map_location="cpu")
    opt_state_dict = torch.load(os.path.join(dirname,latest_opt_ckpt),map_location="cpu")
    
    print("checkpoints '"+latest_model_ckpt+"', '"+latest_opt_ckpt+"'for epoch "+str(latest_epoch)+" have been loaded!")
    return model_state_dict, opt_state_dict, latest_epoch

def load_checkpoint(dirname: str, file_list: str, extension: str):
    # get latest checkpoint
    epochs = [i.split('_', -1)[-1] for i in file_list]
    epochs = [int(i.split('.', -1)[0]) for i in epochs]
    latest_epoch = max(epochs)    
    latest_substring = '_'+str(latest_epoch)+extension
    latest_ckpts = [latest_substring in d for d in file_list]
    temp = np.array(file_list)
    latest_ckpt_files = temp[latest_ckpts]

    try:
        assert len(latest_ckpt_files)==3
    except AssertionError:
        sys.exit("There exist either too many checkpoint-files or one checkpoint-file is missing!")

    model_idx = np.array(["model" in f for f in latest_ckpt_files])
    latest_model_ckpt = latest_ckpt_files[model_idx][0]
    opt_idx = np.array(["optimizer" in f for f in latest_ckpt_files])
    latest_opt_ckpt = latest_ckpt_files[opt_idx][0]
    sched_idx = np.array(["scheduler" in f for f in latest_ckpt_files])
    latest_sched_ckpt = latest_ckpt_files[sched_idx][0]

    model_state_dict = torch.load(os.path.join(dirname,latest_model_ckpt),map_location="cpu")
    opt_state_dict = torch.load(os.path.join(dirname,latest_opt_ckpt),map_location="cpu")
    sched_state_dict = torch.load(os.path.join(dirname,latest_sched_ckpt),map_location="cpu")
    
    print("checkpoints '"+latest_model_ckpt+"', '"+latest_opt_ckpt+"'and '"+latest_sched_ckpt+"'for epoch "+str(latest_epoch)+" have been loaded!")
    return model_state_dict, opt_state_dict, sched_state_dict, latest_epoch


def write_audio_summary(writer: SummaryWriter, mix_audio: Tensor, enhanced_audio: Tensor, target_audio: Tensor, residual_audio: Tensor, tag: str = '', epoch: int=0):
    n_estimates = enhanced_audio.size(-1)

    #audio needs to be down-mixed to mono for audio_summary with tensorboard ==> pytorch-tensorboard only supports mono-audio
    mix_audio = torch.mean(mix_audio,dim=0,keepdim=True)
    enhanced_audio = torch.mean(enhanced_audio,dim=0,keepdim=True)
    target_audio = torch.mean(target_audio,dim=0,keepdim=True)
    residual_audio = torch.mean(residual_audio,dim=0,keepdim=True)

    
    for ii in range(n_estimates):
        if ii>=N_TARGETS:
            writer.add_audio(tag+'/separated: residual', torch.clamp(enhanced_audio[...,ii], max=1), sample_rate=FS, global_step=epoch)

        else:
            writer.add_audio(tag+'/separated: '+TARGET_STR[ii], torch.clamp(enhanced_audio[...,ii], max=1), sample_rate=FS, global_step=epoch)
            writer.add_audio(tag+'/target: '+TARGET_STR[ii], torch.clamp(target_audio, max=1), sample_rate=FS, global_step=epoch)



    writer.add_audio(tag+'/mix', torch.clamp(mix_audio, max=1), sample_rate=FS, global_step=epoch)
    writer.add_audio(tag+'/target: residual', torch.clamp(residual_audio, max=1), sample_rate=FS, global_step=epoch)


    writer.close()


def write_loss_summary(writer: SummaryWriter, overall_loss: Tensor, pre_tag: str, step: int):
    # helper function to add computed loss to tensorboard.

    writer.add_scalar(pre_tag+'/loss/', overall_loss, step)
    writer.close()

def write_metric_summary(writer: SummaryWriter, metrics:Tensor, tag: str, valid_epoch: int):
    
    n_estimates = metrics.size(-1)

    for ii in range(n_estimates):
        if ii>=N_TARGETS:
            #last entry is residual if it is computed!
            writer.add_scalar(tag+'/residual/SDR in dB', metrics[0,ii], valid_epoch)
            writer.add_scalar(tag+'/residual/ISR in dB', metrics[1,ii], valid_epoch)
            writer.add_scalar(tag+'/residual/SIR in dB', metrics[2,ii], valid_epoch)
            writer.add_scalar(tag+'/residual/SAR in dB', metrics[3,ii], valid_epoch)
        else:
            writer.add_scalar(tag+'/'+TARGET_STR[ii]+'/SDR in dB', metrics[0,ii], valid_epoch)
            writer.add_scalar(tag+'/'+TARGET_STR[ii]+'/ISR in dB', metrics[1,ii], valid_epoch)
            writer.add_scalar(tag+'/'+TARGET_STR[ii]+'/SIR in dB', metrics[2,ii], valid_epoch)
            writer.add_scalar(tag+'/'+TARGET_STR[ii]+'/SAR in dB', metrics[3,ii], valid_epoch)

    writer.close()

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


def cleanup(*args):
    # helper function which clears all handed arguments and empties cache of GPU
    for arg in args:
        del arg
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

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

def run_valid_epoch(open_unmx_model: OpenUnmix, encoder: torch.nn.Sequential, decoder: torch.nn.Sequential, dl_valid: DataLoader, loss: torch.nn.MSELoss, compute_metrics: bool, valid_epoch: int):
    
    device = next(open_unmx_model.parameters()).device

    n_batches = len(dl_valid)
    valid_scores = torch.zeros((4,2,n_batches))# tensor for valid-scores shape: [number of valid-metrics x number of estimates x number of batches]
    loss_valid = []
    for valid_batch_ct, valid_batches in enumerate(pbar:=tqdm.tqdm(dl_valid)):

        mix_batch = valid_batches[0].to(device)
        target_batch = valid_batches[2].to(device)

        mix_batch_stft = encoder[0](mix_batch)
        mix_batch_spec = encoder[1](mix_batch_stft)

        estimates_batch_spec = open_unmx_model(mix_batch_spec)

        # calculate loss
        target_batch_spec = encoder(target_batch)
        err = loss(target_batch_spec, estimates_batch_spec.permute(0,3,2,1))
        loss_valid.append(err.detach().cpu().numpy())
    
        if compute_metrics:
            # permute mix and estimates stft-batch into corect order for potential multichannel Wiener filtering
            mix_batch_stft = mix_batch_stft.permute(0,3,2,1,4)
            estimates_batch_spec = estimates_batch_spec.unsqueeze(-1) #[shape=(nb_frames, nb_bins, nb_channels, nb_sources)].

            if USE_WIENER:#(nb_frames, nb_bins, nb_channels, nb_sources)
                estimates_batch_stft = Parallel(n_jobs = NUM_PROCESSES_VALID_LOADER)(delayed(apply_wiener_filter)(estimates_batch_spec[ii,:].detach().cpu(), mix_batch_stft[ii,:].detach().cpu()) for ii in range(BATCH_SIZE_VALID))
                estimates_batch_stft = torch.stack(estimates_batch_stft)
            else:
                # otherwise, we just multiply the targets spectrograms with mix phase
                # we tacitly assume that we have magnitude estimates.
                angle = atan2(mix_batch_stft[..., 1], mix_batch_stft[..., 0])[..., None]

                y = torch.zeros(
                    mix_batch_stft.shape, dtype=mix_batch_stft.dtype, device=mix_batch_stft.device
                )
                y[..., 0] = (estimates_batch_spec * torch.cos(angle)).squeeze()
                y[..., 1] = (estimates_batch_spec * torch.sin(angle)).squeeze()           
                est_residual_stft = mix_batch_stft-y
                estimates_batch_stft = torch.stack((y,est_residual_stft),-1)


            n_samples, n_channels, length = mix_batch.size()
            n_estimates = estimates_batch_stft.size(-1)
            estimate_batch = torch.zeros((n_samples, n_channels, length, n_estimates))

            residual_batch = mix_batch-target_batch

            for ii in range(N_TARGETS+int(COMPUTE_RESIDUAL)):
                estimate_batch[..., ii] = decoder[0](estimates_batch_stft[..., ii].permute(0,3,2,1,4), mix_batch.shape[-1])

            if (valid_batch_ct == 0) or (valid_batch_ct == int(n_batches/2)) or (valid_batch_ct == n_batches-1):
                write_audio_summary(log_writer, mix_batch[0], estimate_batch[0], target_batch[0], residual_batch[0], tag='validation/batch #'+str(valid_batch_ct), epoch=valid_epoch)

            valid_scores[...,valid_batch_ct] = get_BSS_metrics(torch.stack((target_batch, residual_batch), dim=-1), estimate_batch, NUM_PROCESSES_VALID_LOADER)

            cleanup(estimates_batch_stft,estimate_batch)
        else:
            cleanup(estimates_batch_spec)

    if compute_metrics:
        write_metric_summary(log_writer, torch.mean(valid_scores,dim=-1), 'validation-metrics', valid_epoch)
    
    write_loss_summary(log_writer, np.mean(loss_valid), 'mean-validation-loss per epoch', valid_epoch)

    return np.mean(loss_valid)

def apply_wiener_filter(estimates_stft,mixture_stft):
    
    n_frames, n_bins, n_channels, n_complex = mixture_stft.size()
    n_estimates = N_TARGETS + int(COMPUTE_RESIDUAL==True) # if residual is computed ==> there is one more estimate-array
    estimates_out_stft = torch.zeros((n_frames, n_bins, n_channels, n_complex, n_estimates)).to(estimates_stft.device)

    pos = 0
    if WIENER_LEN>0:
        wiener_win_len = WIENER_LEN
    else:
        wiener_win_len = n_frames
    while pos < n_frames:
        cur_frame = torch.arange(pos, min(n_frames, pos + wiener_win_len))
        pos = int(cur_frame[-1]) + 1

        estimates_out_stft[cur_frame] = wiener(
            estimates_stft[cur_frame],
            mixture_stft[cur_frame],
            N_WIENER_ITs,
            softmask=WIENER_SOFTMAX_INIT,
            residual=COMPUTE_RESIDUAL,
        )

        # targets_spectrograms (Tensor): spectrograms of the sources
        #     [shape=(nb_frames, nb_bins, nb_channels, nb_sources)].
        #     This is a nonnegative tensor that is
        #     usually the output of the actual separation method of the user. The
        #     spectrograms may be mono, but they need to be 4-dimensional in all
        #     cases.
        # mix_stft (Tensor): [shape=(nb_frames, nb_bins, nb_channels, complex=2)]
        #     STFT of the mixture signal.

    return estimates_out_stft



def main():
    print("Start training of open-unmix model!")

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    musdb_foldername_list = np.array(get_subfoldernames_in_folder(TRAIN_FILEPATH))
    musdb_folder_ids = np.arange(len(musdb_foldername_list))

    rand_train_ids = np.random.randint(0,len(musdb_foldername_list), int(TRAIN_DATA_PERCENTAGE*len(musdb_foldername_list)))
    rand_train_ids_unique = np.in1d(musdb_folder_ids,rand_train_ids)
    valid_id_pool = musdb_folder_ids[np.invert(rand_train_ids_unique)]
    rand_valid_ids = np.random.choice(valid_id_pool,round((1-TRAIN_DATA_PERCENTAGE)*len(musdb_foldername_list)))#p.random.randint(0,len(valid_id_pool), int((1-TRAIN_DATA_PERCENTAGE)*len(valid_id_pool)))
    train_tracks_foldername_list = list(musdb_foldername_list[rand_train_ids].astype(list))
    valid_tracks_foldername_list = list(musdb_foldername_list[rand_valid_ids].astype(list))

    torchaudio.set_audio_backend(config['audio_backend'])

    mus_train = MSSMUSDBDataset(train_tracks_foldername_list,
                                target_str = ['bass','vocals','other','drums'],
                                random_mix_flag = RANDOM_MIX_FLAG,
                                augmentation_flag = True,
                                duration = 10,
                                samples_per_track = SAMPLES_PER_TRACK_TRAIN,
                                fs = 44100
                                )

    mus_valid = MSSMUSDBDataset(valid_tracks_foldername_list,
                            target_str = ['bass','vocals','other','drums'],
                            random_mix_flag = False,
                            augmentation_flag = False,
                            duration = 10,
                            samples_per_track = SAMPLES_PER_TRACK_VALID,
                            fs = 44100
                            )


    dl_train = DataLoader(mus_train, BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_PROCESSES_TRAIN_LOADER,
        pin_memory=True, worker_init_fn = worker_init_fn_rand, drop_last = False, persistent_workers=True)

    dl_valid = DataLoader(mus_valid, BATCH_SIZE_VALID, shuffle=False, num_workers=NUM_PROCESSES_VALID_LOADER, worker_init_fn = numpy_fixed_seed, drop_last = False)

    train_device = torch.device("cuda")#torch.device("cpu")


    stft, istft = make_filterbanks(n_fft=NFFT, n_hop=NHOP, sample_rate=FS, center=True, length = int(FS*SEQ_DUR_TRAIN))
    encoder = torch.nn.Sequential(stft, ComplexNorm(mono= NB_CHANNELS == 1))
    decoder = torch.nn.Sequential(istft)


    train_dataset_mu, train_dataset_std = get_statistics(encoder, mus_train, VERBOSE, USE_LOG_FEATURES)
    max_bin = bandwidth_to_max_bin(FS, NFFT, BANDWIDTH)

    if USE_LOG_FEATURES:
        open_unmx_model = OpenUnmixLog(nb_bins=NBINS, nb_channels=NB_CHANNELS, hidden_size=HIDDEN_SIZE, max_bin = max_bin, nb_layers=N_LAYERS, unidirectional=UNI_DIR_FLG, input_mean=train_dataset_mu, input_scale=train_dataset_std)
    else:
        open_unmx_model = OpenUnmix(nb_bins=NBINS, nb_channels=NB_CHANNELS, hidden_size=HIDDEN_SIZE, max_bin = max_bin, nb_layers=N_LAYERS, unidirectional=UNI_DIR_FLG, input_mean=train_dataset_mu, input_scale=train_dataset_std)
    
    open_unmx_model.to(train_device)
    open_unmx_model.train()
    optimizer = torch.optim.Adam(open_unmx_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=LR_DECAY_GAMMA,
            patience=LR_DECAY_PATIENCE,
            cooldown=10,
        )

    os.makedirs(CKPT_PATH, exist_ok=True)
    ckpt_file_list = os.listdir(CKPT_PATH)

    if len(ckpt_file_list) >= 2:
        # if at least two checkpoint files (model and optimizer) exist => load checkpoints
        if USE_LR_SCHEDULER:
            latest_model_ckpt, latest_opt_ckpt, latest_sched_ckpt, start_epoch = load_checkpoint(CKPT_PATH, ckpt_file_list, ".ckpt")
            open_unmx_model.load_state_dict(latest_model_ckpt)
            optimizer.load_state_dict(latest_opt_ckpt)
            scheduler.load_state_dict(latest_sched_ckpt)
        else:
            latest_model_ckpt, latest_opt_ckpt, start_epoch = load_checkpoint_no_scheduler(CKPT_PATH, ckpt_file_list, ".ckpt")
            open_unmx_model.load_state_dict(latest_model_ckpt)
            optimizer.load_state_dict(latest_opt_ckpt)
        start_epoch += 1
    else:
        # no checkpoint files are found => create new ones at given directory path
        print('No checkpoints found! Training starts with epoch #0!')
        start_epoch = 0

    # if START_w_VALID flag is set => before training a validation epoch is started
    global START_w_VALID
    if START_w_VALID:
        valid_epoch = start_epoch-1 
    else:
        valid_epoch = start_epoch


    get_model_summary(FS,open_unmx_model,train_device)

    loss = torch.nn.MSELoss()
    loss_over_epochs = [None]*start_epoch
    loss_over_batches = []
    loss_list_temp = []
    batch_log_ct = 0
    N_log_batches = 80

    for epoch in range(MAX_EPOCHS)[start_epoch:]:

        if START_w_VALID:
            #if flag is set to True => start with validation
            open_unmx_model.eval()
            run_valid_epoch(open_unmx_model, encoder, decoder, dl_valid, loss, True, valid_epoch)
            valid_epoch += 1
            START_w_VALID = False

        
        open_unmx_model.train()
        for batch_ct, batches in enumerate(pbar:=tqdm.tqdm(dl_train)):
            
            mix_batch = batches[0].to(train_device)
            target_batch = batches[2].to(train_device)

            optimizer.zero_grad()
            mix_batch_stft = encoder[0](mix_batch).to(train_device)
            mix_batch_spec = encoder[1](mix_batch_stft).to(train_device)

            target_batch_stft = encoder[0](target_batch).to(train_device)
            target_batch_spec = encoder[1](target_batch_stft).to(train_device)

            enh_batch_spec = open_unmx_model(mix_batch_spec)

            # calculate loss
            err = loss(target_batch_spec, enh_batch_spec.permute(0,3,2,1))
            loss_over_batches.append(err.detach().cpu().numpy())

            # execute backward-propagation
            err.backward()
            optimizer.step()

            # index for summary sample if 0 => first sample of exemplary batches in validation are used
            summary_sample_idx = 0

            # write summarys and command window logs every N_log_epochs batches
            if (batch_ct+1)%N_log_batches == 0 or batch_ct==0:
                if batch_ct>0:
                    batch_log_ct += N_log_batches

                    if USE_WIENER:#(nb_frames, nb_bins, nb_channels, nb_sources)
                        # permute mix and estimates stft-batch into corect order for potential multichannel Wiener filtering
                        mix_batch_stft = mix_batch_stft.permute(0,3,2,1,4)
                        enh_batch_spec = enh_batch_spec.unsqueeze(-1)
                        log_batch_stft = apply_wiener_filter(enh_batch_spec[summary_sample_idx,:], mix_batch_stft[summary_sample_idx,:])
                    else:
                        # use phase of mixture for reconstruction of enhanced batch.
                        mix_stft = mix_batch_stft.permute(0,3,2,1,4).to("cpu")
                        angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]
                        y = torch.zeros(
                            mix_stft.shape, dtype=mix_stft.dtype, device=mix_stft.device
                        )
                        y[..., 0] = (enh_batch_spec.unsqueeze(-1).to("cpu") * torch.cos(angle)).squeeze()
                        y[..., 1] = (enh_batch_spec.unsqueeze(-1).to("cpu") * torch.sin(angle)).squeeze()           
                        est_residual_stft = mix_stft-y
                        log_batch_stft = torch.stack((y,est_residual_stft),-1)

                    n_samples, n_channels, length = mix_batch.size()
                    n_estimates = log_batch_stft.size(-1)
                    estimate_batch = torch.zeros((n_samples, n_channels, length, n_estimates))

                    residual_batch = (mix_batch-target_batch).to("cpu")

                    for ii in range(N_TARGETS+int(COMPUTE_RESIDUAL)):
                        estimate_batch[..., ii] = decoder[0](log_batch_stft[..., ii].permute(0,3,2,1,4), mix_batch.shape[-1])

                    write_audio_summary(log_writer, mix_batch[summary_sample_idx], estimate_batch[summary_sample_idx], target_batch[summary_sample_idx], residual_batch[summary_sample_idx], tag='training/batch #'+str(batch_log_ct), epoch=batch_log_ct)
                    loss_list_temp.append(np.mean(loss_over_batches))
                    loss_over_batches = []
                    batch_log_ct += N_log_batches
            pbar.set_description('training-epoch #'+str(epoch)+', loss: '+str(err.detach().cpu().numpy()))
        # write mean loss of all training batches to list containing loss per epoch
         
        loss_over_epochs.append(np.mean(loss_list_temp))
        write_loss_summary(log_writer, loss_over_epochs[epoch], 'mean-training-loss per epoch', epoch)

        cleanup(mix_batch, target_batch, mix_batch_stft, mix_batch_spec, target_batch_stft, target_batch_spec, enh_batch_spec)

        batch_log_ct = 0
            
        open_unmx_model.eval()
        if ((epoch+1)%N_LOG_EPOCHS == 0) or (epoch==0):
            # run validation epoch and compute metrics for every N_log_metrics_valid
            valid_loss = run_valid_epoch(open_unmx_model, encoder, decoder, dl_valid, loss, True, valid_epoch)
        else:
            # run validation epoch and only compute loss
            valid_loss = run_valid_epoch(open_unmx_model, encoder, decoder, dl_valid, loss, False, valid_epoch)
        
        # write checkpoints and logs!
        if USE_LR_SCHEDULER:
            scheduler.step(valid_loss)

        if ((epoch+1)%N_LOG_EPOCHS==0):           
            write_checkpoint(open_unmx_model, 'Open_Unmix_model_', CKPT_PATH, epoch, "ckpt")
            write_checkpoint(optimizer, 'Open_Unmix_optimizer_', CKPT_PATH, epoch, "ckpt")
            if USE_LR_SCHEDULER:
                write_checkpoint(scheduler, 'Open_Unmix_scheduler_', CKPT_PATH, epoch, "ckpt")

        #set model to training mode for next epoch
        open_unmx_model.train()

        valid_epoch += 1
        # write checkpoints and logs! 

        #empty GPU cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         


    print("This is the end. My only friend, the end. -Jim Morrison, John Paul Densmore, Robert Krieger & Raymond Manzarek (The Doors), 1967")



if __name__ == "__main__":
    main()

