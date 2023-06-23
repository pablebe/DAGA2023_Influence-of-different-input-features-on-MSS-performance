import os
import soundfile
import numpy as np
import torch
import webrtcvad
from torch.utils import data

def compute_speech_rms(x_segment):
    """ calculate rms value from list of speech segments """
    speech_rms = 0.0
    speech_len = 0
    speech_len = x_segment.size(1)
    speech_rms = torch.sum(x_segment**2,1)
    speech_rms = torch.sqrt(speech_rms/speech_len)
    return speech_rms

def augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio


class MSSMUSDBDataset(data.Dataset):
    def __init__(self, foldername_list, target_str, random_mix_flag, augmentation_flag, duration, samples_per_track, fs) -> None:
        super().__init__()

        self.foldername_list = foldername_list
        self.target_str = target_str
        self.random_mixing = random_mix_flag
        self.augmentations = augmentation_flag
        self.duration = duration
        self.fs = fs
        self.length = len(foldername_list)
        self.samples_per_track = samples_per_track

    def __len__(self):
        return self.length*self.samples_per_track

    def get_training_sample(self, mixture_filepath, target_bass_filepath, target_vocals_filepath, target_other_filepath, target_drums_filepath, random_mixing_flag, augmentation_flag, duration, fs):

        seq_dur = int(np.floor(fs*duration))

        if random_mixing_flag:
            bass_info = soundfile.info(target_bass_filepath)
            bass_len = bass_info.frames
            bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)

            vocals_info = soundfile.info(target_vocals_filepath)
            vocals_len = vocals_info.frames
            vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)

            other_info = soundfile.info(target_other_filepath)
            other_len = other_info.frames
            other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)

            drums_info = soundfile.info(target_drums_filepath)
            drums_len = drums_info.frames
            drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
        else:
                    
            mixture_info = soundfile.info(mixture_filepath)
            mixture_len = mixture_info.frames

            mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
#            mixture, _ = soundfile.read(mixture_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')

            bass_start_index = mixture_start_index
            vocals_start_index = mixture_start_index
            other_start_index = mixture_start_index
            drums_start_index = mixture_start_index


        target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
        target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
        target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
        target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')


        if augmentation_flag:
            audio_bass = torch.as_tensor(target_bass.T, dtype=torch.float32)
            audio_bass = augment_force_stereo(audio_bass)
            audio_bass = augment_gain(audio_bass)
            audio_bass = augment_channelswap(audio_bass)
            target_bass = audio_bass

            audio_vocals = torch.as_tensor(target_vocals.T, dtype=torch.float32)
            audio_vocals = augment_force_stereo(audio_vocals)
            audio_vocals = augment_gain(audio_vocals)
            audio_vocals = augment_channelswap(audio_vocals)
            target_vocals = audio_vocals

            audio_other = torch.as_tensor(target_other.T, dtype=torch.float32)
            audio_other = augment_force_stereo(audio_other)
            audio_other = augment_gain(audio_other)
            audio_other = augment_channelswap(audio_other)
            target_other = audio_other

            audio_drums = torch.as_tensor(target_drums.T, dtype=torch.float32)
            audio_drums = augment_force_stereo(audio_drums)
            audio_drums = augment_gain(audio_drums)
            audio_drums = augment_channelswap(audio_drums)
            target_drums = audio_drums

        else:
            target_bass = torch.Tensor(target_bass.T)
            target_vocals = torch.Tensor(target_vocals.T)
            target_other = torch.Tensor(target_other.T)
            target_drums = torch.Tensor(target_drums.T)

        mixture = torch.stack((target_bass, target_vocals, target_other, target_drums),-1).sum(-1)

        vocal_rms = compute_speech_rms(target_vocals)
        vocal_rms[0] = max((vocal_rms[0], 10**(-60/20)))
        vocal_rms[1] = max((vocal_rms[1], 10**(-60/20)))



        return torch.Tensor(mixture), torch.Tensor(target_bass), torch.Tensor(target_vocals), torch.Tensor(target_other), torch.Tensor(target_drums), vocal_rms#, mixture_filepath, mixture_start_index#x, y_bass, y_vocals, y_other, y_drums

    def __getitem__(self, item):
        mixture_track_folder = self.foldername_list[item//self.samples_per_track]

        if self.random_mixing:      
            item_bass = np.random.randint(self.__len__())//self.samples_per_track
            while item_bass == item//self.samples_per_track:
                item_bass = np.random.randint(self.__len__())//self.samples_per_track

            item_vocals = np.random.randint(self.__len__())//self.samples_per_track
            while item_vocals == item//self.samples_per_track:
                item_vocals = np.random.randint(self.__len__())//self.samples_per_track

            item_other = np.random.randint(self.__len__())//self.samples_per_track
            while item_other == item//self.samples_per_track:
                item_other = np.random.randint(self.__len__())//self.samples_per_track

            item_drums = np.random.randint(self.__len__())//self.samples_per_track
            while item_drums == item//self.samples_per_track:
                item_drums = np.random.randint(self.__len__())//self.samples_per_track
             
            track_folder_bass = self.foldername_list[item_bass]
            track_folder_vocals =self.foldername_list[item_vocals]
            track_folder_other = self.foldername_list[item_other]
            track_folder_drums = self.foldername_list[item_drums]
        else:
            track_folder_bass = self.foldername_list[item//self.samples_per_track]
            track_folder_vocals =self.foldername_list[item//self.samples_per_track]
            track_folder_other = self.foldername_list[item//self.samples_per_track]
            track_folder_drums = self.foldername_list[item//self.samples_per_track]

        mixture_filepath = os.path.join(mixture_track_folder,'mixture.wav')
        target_bass_filepath = os.path.join(track_folder_bass, 'bass.wav')
        target_vocals_filepath = os.path.join(track_folder_vocals,'vocals.wav')
        target_other_filepath = os.path.join(track_folder_other,'other.wav')
        target_drums_filepath = os.path.join(track_folder_drums,'drums.wav')

        temp = self.get_training_sample(mixture_filepath, target_bass_filepath, target_vocals_filepath, target_other_filepath, target_drums_filepath, self.random_mixing, self.augmentations, self.duration, self.fs)

        
        return temp
