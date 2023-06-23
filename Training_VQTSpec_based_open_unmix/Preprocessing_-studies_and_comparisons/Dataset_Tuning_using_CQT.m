%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab script to tune dataset to a tuning-frequency. Can be useful when
% working with NSGT-like transforms in order to tune center frequencies of 
% filterbank. 
%
% For tuning the blockbased CQT algorithm from [1] is used. Matlab script
% was adapted from [2]. 
% The code for [2] can be found here: 
% https://git.iem.at/audioplugins/cqt-analyzer
%
%
% ADA - blockbased CQT, Source: Schörkhuber [1]
%
% [1] Ch. Schörkhuber et al., Toolbox for Efficient Perfect Reconstruction
% Time-Frequency Transforms with Log-Frequency Resolution, AES 53rd Int.
% Conf., London, UK, 2014.
%
% [2] F. Holzmüller, P. Bereuter, P. Merz, D. Rudrich and A. Sontacchi,  
%  Computational efficient real-time capable constant-Q spectrum analyzer, 
%  in Proceedings of the AES 148th Convention, May 2020, [Online]. 
%  Available: http://www.aes.org/e-lib/browse.cfm?elib=20805.
%
% READ ME:
% In order for this code to work the Folder 'RUMS' has to be added to the
% matlab path with all subfolders.
% 'RUMS' can be found here: https://git.iem.at/rudrich/rums
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear variables;
close all;
clc;


set(0,'defaulttextInterpreter','latex')
set(0,'defaultlegendInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex')


fs = 44100;

% USER PARAMETERSETTINGS

fmin = 55;   % lowest frequency which is analyzed
NOctaves = 5; % Number of analyzed octaves
B = 72;       % Bins per octave e.g B=12 => semitone-resolution
gamma = 0;    % increases Bandwidth for lower frequencies => decreases Q


%% internal parameters derived from User-Settings
overSamplingFactor = 2;
Q = (2^(1 / B) - 2^(-1 / B))^-1; %Q-value for constant Q case
alpha = 1/Q;

fmax = fmin * 2^NOctaves; % highest frequency which is to be analyzed
K = log2(fmax/fmin)*B;  % number of analyzed bins

fm = @(fmin, k, B) 2^(1 / B).^k * fmin; % center-frequency calculation

fk = fm (fmin, [0:K-1], B); % calculate center-frequencies

% apply gamma-value
Bk = alpha.*fk+gamma; % new bandwidths are calculated
Qnew = fk./Bk; % Q-value with gamma decreases for lower-frequencies

Nk = round(fs./fmin.*Qnew);
N_max = max(Nk); % maximum window-size

b_new = log(2)./asinh(0.5./Qnew); % new CQT-resolution in bins per octave

L = 2^nextpow2(N_max);
NFFT  = overSamplingFactor * L;

df = fs / NFFT; % Frequency-resolution of big FFT

clear Nk_max
%% number of frequency-points in k-subbands: start-, center-, stop-bin

% calculate start and stop-bins of Windows in frequency domain.
% start-/stop-bins are calculated by subtracting/adding half of the
% corresponding bandwidth from/to the center-bin
% Not completely correct => there are more frequency-bins in the upper-half
% of the window than in the lower half!

% Bk = floor(Bnew/df);

fBinStart = ceil((fk.*2.^(-1./b_new))./df)+1; 
fBinStop = floor((fk.*2.^(1./b_new))./df)+1;
fBinCenter = round(fk./df);

fStart = fk.*2.^(-1./b_new);
fStop = fk.*2.^(1./b_new);

fBinStart(fBinStart<=0)=1;

Bk = fBinStop - fBinStart + 1;

%To see a demonstration of the impact of gamma set gammaDemoFlag to 1
gammaDemoFlag = 0;
if gammaDemoFlag
    % values at the end are gamma-values [0, 3, 10, 24.7/0.108*alpha, 30]
    Bdemo(1,:) = alpha.*fk+0;
    Bdemo(2,:) = alpha.*fk+3;
    Bdemo(3,:) = alpha.*fk+10;
    Bdemo(4,:) = alpha.*fk+24.7/0.108*alpha;
    Bdemo(5,:) = alpha.*fk+30;
    % for frequencies above 500 Hz => appro. consant Q!
    figure
    loglog(fk,Bdemo,'LineWidth',1)
    grid on
    xlim([10 10^4])
    ylim([1 10^3])
    legend('$\gamma = 0$','$\gamma = 3$','$\gamma = 10$',...
           '$\gamma = \Gamma$', '$\gamma = 30$')
    title('Bandwidths for different $\gamma$')
    xlabel('frequency in [Hz]')
    ylabel('Bandwith B in [Hz]')
    axis tight

    figure
    plot(fk,Qnew)
    xlabel('frequency in [Hz]')
    ylabel('Q-value')
    title('Q over center frequencies')
    grid on;

end

%% calculation of IDFT-length and Overlap

Bkmax = max(Bk);       % max. bandwidth - highest subband
M = 2^nextpow2(Bkmax); % IDFT-length

% proportionality-factor between original fs and subband fs,k (k=kmax)
divFact = NFFT/M;

%% calculation of window-functions
% A window with quite high resolution is calculated as lookup-window. The
% lookup-window is scanned for the closest freqencies in respect to the FFT
% bin-freuencies. These values are now taken for the windows in
% freq-domain. So an interpolated, warped window can be designed quite
% easily.


% the windows in frequency-domain are written into the matrix W
W = zeros(NFFT/2+1,K);

w_lookup = hann(NFFT/2 + 1, 'periodic').';
halfwin_len = floor(length(w_lookup)/2);

fft_freqs = 0:df:fs/2;

for k = 1:K
    f_win = fk(k) * 2.^((-halfwin_len:halfwin_len)/(b_new(k) * halfwin_len));

    for ii = fBinStart(k):fBinStop(k)
        [~, nearestBin] = min(abs(f_win - fft_freqs(ii)));
        W(ii, k) = w_lookup(nearestBin);
    end
end



%% import audio-data and determine sampling frequency fs

%MUSDB-dataset-path:
% ENTER PATH TO DATASET AND OUTPUT FOLDER!
musdb_path = './musdb18hq/';
out_path = './musdb18hq/tuned_to_440_Hz';
TRAIN_SET = false;

if TRAIN_SET
    filepath = strcat(musdb_path,'/train');
    filepath_out = strcat(out_path,'/train');
else
    filepath = strcat(musdb_path,'/test');
    filepath_out = strcat(out_path,'/test');

end

dir_struct = dir(filepath);
folders = {dir_struct.folder};
names = {dir_struct.name};
idx = not(strcmp(names,'.') | strcmp(names,'..') | strcmp(names,'.DS_Store'));

folders = folders(idx);
names = names(idx);

N_files = length(folders);
NIts = 1;

tuning_log = cell(2,N_files,4);

for tuningIts = 1:NIts
    disp(strcat('tuning-iteration #',num2str(tuningIts)));
    if tuningIts>1
        musdb_path = '/Users/Paul/IEM-Phd/03_PhD/01_Datasets/musdb18hq/tuned_to_440_Hz';
        if TRAIN_SET
            filepath = strcat(musdb_path,'/train');
            filepath_out = strcat(out_path,'/train');
        else
            filepath = strcat(musdb_path,'/test');
            filepath_out = strcat(out_path,'/test');
        
        end
    end

for file_id=1:N_files
   disp(strcat('file #',num2str(file_id)))
pathName = string(folders(file_id));
fileName = string(names(file_id));

x_vox = AudioData (strcat(pathName,'/',fileName,'/vocals.wav')); % open audio-file
x_mix = AudioData (strcat(pathName,'/',fileName,'/mixture.wav')); % open audio-file
x_drums = AudioData (strcat(pathName,'/',fileName,'/drums.wav')); % open audio-file
x_other = AudioData (strcat(pathName,'/',fileName,'/other.wav')); % open audio-file
x_bass = AudioData (strcat(pathName,'/',fileName,'/bass.wav')); % open audio-file

AudioData (string(strcat(pathName,'/',fileName,'/vocals.wav'))); % open audio-file

x_vox_mono = copy(x_vox);
x_vox_mono.data = mean (x_vox.data, 2); % convert to mono
%t = (0:x_vox_mono.duration/(x_vox_mono.numSamples-1):x_vox_mono.duration).';
%x_vox_mono.data = sin(2*pi*444*t);
%x_vox.data = sin(2*pi*444*t);
[detune_Fact,fTune_mean] = calc_detuning(x_vox_mono,L,overSamplingFactor,M,K,W,k,fBinStart,fBinStop,fk,B);

tuning_log{tuningIts,file_id,1} = fTune_mean;
tuning_log{tuningIts,file_id,2} = detune_Fact;
tuning_log{tuningIts,file_id,3} = size(x_vox.data,1);

%if detune_Fact<1
q = 10000;
p = round(detune_Fact*q);
% else
%     p = 10000;
%     q = round(detune_Fact*p);
% end

audio_vox_resample = resample(x_vox.data,p,q);
audio_vox_resample(audio_vox_resample>1) = 1;
audio_vox_resample(audio_vox_resample<-1) = -1;
audio_bass_resample = resample(x_bass.data,p,q);
audio_bass_resample(audio_bass_resample>1) = 1;
audio_bass_resample(audio_bass_resample<-1) = -1;
audio_other_resample = resample(x_other.data,p,q);
audio_other_resample(audio_other_resample>1) = 1;
audio_other_resample(audio_other_resample<-1) = -1;
audio_drums_resample = resample(x_drums.data,p,q);
audio_drums_resample(audio_drums_resample>1) = 1;
audio_drums_resample(audio_drums_resample<-1) = -1;
audio_mixture_resample = resample(x_mix.data,p,q);
audio_mixture_resample(audio_mixture_resample>1) = 1;
audio_mixture_resample(audio_mixture_resample<-1) = -1;

tuning_log{tuningIts,file_id,4}=size(audio_vox_resample,1);

if not(isfolder(filepath_out))
    mkdir(filepath_out)
end
output_folder = strcat(filepath_out,'/',fileName);
if not(isfolder(output_folder))
    mkdir(strcat(output_folder))
end

audiowrite(strcat(output_folder,'/vocals.wav'),audio_vox_resample,fs);
audiowrite(strcat(output_folder,'/bass.wav'),audio_bass_resample,fs);
audiowrite(strcat(output_folder,'/other.wav'),audio_other_resample,fs);
audiowrite(strcat(output_folder,'/drums.wav'),audio_drums_resample,fs);
audiowrite(strcat(output_folder,'/mixture.wav'),audio_mixture_resample,fs);


end
end
if TRAIN_SET
    save('train_set_tuning_log.mat','tuning_log')
else
    save('test_set_tuning_log.mat','tuning_log')
end
DBG = 0;

%% functions

function [detune_Fact,fTune_mean]=calc_detuning(audio_struct,L,overSamplingFactor,M,K,W,k,fBinStart,fBinStop,fk,B)
%% buffering, windowing and zero-padding of imported audio-file
    xBlocks = buffer (audio_struct, L, L / 2, 'nodelay');
    xBlocks.applyWindow (@hann);
    
    % zero-pad the imported audio-file
    numZeroPadSamples = (overSamplingFactor - 1) * L;
    xBlocks.data{1} = [zeros(numZeroPadSamples / 2, xBlocks.numBlocks);...
            xBlocks.data{1}; zeros(numZeroPadSamples / 2, xBlocks.numBlocks)];
    
    % execute a real valued fft (one-sided).
    X = xBlocks.rfft;
    
    %% calculation of CQT-coefficients
    
    % hop size
    hs = M / overSamplingFactor / 2;
    
    C = zeros (M - hs, K); % preallocation of coefficient matrix.
    
    for frame =  1 : X.numBlocks
        % apply windows in the frequency-domain
        Xfilt = X.data{1}(:, frame) .* W;
        ifftData = zeros (M, k);
        % extract values between start- and stop-bins => sub-sampling
        for k = 1 : K
            nSmpls = fBinStop(k) - fBinStart(k) + 1;
            fbData = Xfilt(fBinStart(k) : fBinStop(k), k);
            ifftData(1:nSmpls, k) = fbData;
            % Shift shift to base-band is only necessary if icqt is planned
            % => not needed for Analysis-purposes
            % ifftData(:,k) = circshift(ifftData(:,k),-round(nSmpls/2));
        end
        c = abs (ifft (ifftData)); % perform IDFT on the subsampled values
        % overlap and add
        C(end - (M - hs) + 1 : end, :) = C(end - (M - hs) + 1 : end, :) +...
                                         c(1 : M - hs, :);
        C = [C; c(M - hs + 1 : end, :)];
    end
    
    %% Tuner as implemented in the plugin:
    % Show deviation in cent to tuning frequency.
    fTune = 440;
    
    NearestBins = zeros(size(C,1),1);
    
    % get nearest bin to tuning frequency.
    minOffset = 100;
    for k=1:K
        if abs(fk(k) - fTune) < minOffset
            tuningBinOrig = k;
            minOffset = abs(fk(k)-fTune);
        end
    end
    tuningBin = tuningBinOrig;
    
    BinsPerSemiTone = B/12;
    CSum = zeros(size(C,1),3);
    
    nn=1;
    % while loop to replace goto command in JUCE
    while nn<size(C,1)
     % matlab index start with 1 therefore tuningBin - 1 to ensure modulo
     % operator works
     modTuning = mod(tuningBin-1,BinsPerSemiTone);
       % Sum Up all corresponding semitone-candidates into 3 bins,  e.g.
       % Sum up every fourth bin for B=48
       for ii = 0:K-1
           if mod(ii,BinsPerSemiTone) == modTuning
            CSum(nn,2)=CSum(nn,2)+C(nn,ii+1);
           end
           if mod(ii,BinsPerSemiTone) == mod((modTuning + 1),BinsPerSemiTone)
            CSum(nn,3)=CSum(nn,3)+C(nn,ii+1);
           end
           if mod(ii,BinsPerSemiTone) == mod((modTuning - 1 + BinsPerSemiTone),BinsPerSemiTone)
            CSum(nn,1)=CSum(nn,1)+C(nn,ii+1);
           end
       end
       % Shift if maximum energy doesn't lie in center-candidate and
       % start calculation again => in JUCE this is done with "goto"
       if CSum(nn,1)>CSum(nn,2)
            tuningBin = tuningBin-1;
            CSum(nn,1:3)=0;
       elseif CSum(nn,3)>CSum(nn,2)
            tuningBin = tuningBin+1;
            CSum(nn,1:3)=0;
       else
        % Apply parabolic interpolation with geometric frequency-bin relation
        % Amplitude Values for parabolic interpolation
           fa = CSum(nn,1);
           fb = CSum(nn,2);
           fc = CSum(nn,3);
    
        % Frequency Values for parabolic interpolation
           a = fk(tuningBin-1);
           b = fk(tuningBin);
           c = fk(tuningBin+1);
           % source: http://fourier.eng.hmc.edu/e176/lectures/NM/node25.html
           % calculate parabolically interpolated frequency
           fTuneParab(nn) = b + 1/2*(((fa-fb)*(c-b)^2-((fc-fb)*(b-a)^2))/...
                            ((fa-fb)*(c-b)+(fc-fb)*(b-a)));
    
           nn = nn+1;
           tuningBin = tuningBinOrig;
        end
        nn = max([nn,1]);
    end
    fTune_mean = nanmean(fTuneParab);
    detune_Fact = fTune_mean/fTune;
    

end


