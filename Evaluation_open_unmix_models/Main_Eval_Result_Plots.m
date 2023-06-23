%% Matlab Script to create result plots of [1]
%
%   References:
%   [1] Paul A. Bereuter & Alois Sontacchi, Influence of Different Input 
%       Features on Musical Source Separation Performance; Fortschritte der
%       Akustik - DAGA 2023; p.430-433; 
%       URL: https://pub.dega-akustik.de/DAGA_2023/data/daga23_proceedings.pdf
%
%
% Created by Paul A. Bereuter (March 2023)

close all
clear
clc
% set(0,'defaulttextInterpreter','latex')
% set(0,'defaultlegendInterpreter','latex')
% set(0,'defaultAxesTickLabelInterpreter','latex')

set(0,'defaultAxesFontName','Times New Roman')
set(0,'defaultTextFontName','Times New Roman')
labelangle = 15;
model_1_local_vocal_blocked = readtable("./evaluation_scores/test_score_vocals_model_1.csv");
sdr_model1_blocked = mean(table2array(model_1_local_vocal_blocked(:,"sdr")));


model_1_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_1.csv");
model_2_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_2.csv");
model_3_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_3.csv");
model_4_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_4.csv");
model_5_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_5.csv");
model_6_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_6.csv");
model_7_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_7.csv");
model_8_local_vocal = readtable("./evaluation_scores/test_score_vocals_model_8.csv");

model_1_local_residual = readtable("./evaluation_scores/test_score_residual_model_1.csv");
model_2_local_residual = readtable("./evaluation_scores/test_score_residual_model_2.csv");
model_3_local_residual = readtable("./evaluation_scores/test_score_residual_model_3.csv");
model_4_local_residual = readtable("./evaluation_scores/test_score_residual_model_4.csv");
model_5_local_residual = readtable("./evaluation_scores/test_score_residual_model_5.csv");
model_6_local_residual = readtable("./evaluation_scores/test_score_residual_model_6.csv");
model_7_local_residual = readtable("./evaluation_scores/test_score_residual_model_7.csv");
model_8_local_residual = readtable("./evaluation_scores/test_score_residual_model_8.csv");

noisy_scores = readtable("./evaluation_scores/noisy_scores.csv");
IRM_scores_vocals = readtable("./evaluation_scores/IRM_scores_vocals.csv");
IRM_scores_residual = readtable("./evaluation_scores/IRM_scores_residual.csv");
IRMVQ_scores_vocals = readtable("./evaluation_scores/IRMVQ_scores_vocals.csv");
IRMVQ_scores_residual = readtable("./evaluation_scores/IRMVQ_scores_residual.csv");

sdr_local_model_1_vocals = table2array(model_1_local_vocal(:,"sdr"));
sdr_local_model_2_vocals = table2array(model_2_local_vocal(:,"sdr"));
sdr_local_model_3_vocals = table2array(model_3_local_vocal(:,"sdr"));
sdr_local_model_4_vocals = table2array(model_4_local_vocal(:,"sdr"));
sdr_local_model_5_vocals = table2array(model_5_local_vocal(:,"sdr"));
sdr_local_model_6_vocals = table2array(model_6_local_vocal(:,"sdr"));
sdr_local_model_7_vocals = table2array(model_7_local_vocal(:,"sdr"));
sdr_local_model_8_vocals = table2array(model_8_local_vocal(:,"sdr"));
sdr_noisy = table2array(noisy_scores(:,"sdr"));
sdr_IRM_vocals = table2array(IRM_scores_vocals(:,"sdr"));
sdr_IRMVQ_vocals = table2array(IRMVQ_scores_vocals(:,"sdr"));
ORACLE_SDR = [sdr_noisy,sdr_IRM_vocals, sdr_IRMVQ_vocals];
SDR_vocals = [sdr_local_model_1_vocals,sdr_local_model_2_vocals,sdr_local_model_3_vocals,sdr_local_model_4_vocals,sdr_local_model_5_vocals,sdr_local_model_6_vocals,sdr_local_model_7_vocals,sdr_local_model_8_vocals,sdr_IRM_vocals, sdr_IRMVQ_vocals];

sdr_local_model_1_residual= table2array(model_1_local_residual(:,"sdr"));
sdr_local_model_2_residual = table2array(model_2_local_residual(:,"sdr"));
sdr_local_model_3_residual = table2array(model_3_local_residual(:,"sdr"));
sdr_local_model_4_residual = table2array(model_4_local_residual(:,"sdr"));
sdr_local_model_5_residual = table2array(model_5_local_residual(:,"sdr"));
sdr_local_model_6_residual = table2array(model_6_local_residual(:,"sdr"));
sdr_local_model_7_residual = table2array(model_7_local_residual(:,"sdr"));
sdr_local_model_8_residual = table2array(model_8_local_residual(:,"sdr"));
sdr_IRM_residual = table2array(IRM_scores_residual(:,"sdr"));
sdr_IRMVQ_residual = table2array(IRMVQ_scores_residual(:,"sdr"));

SDR_residual = [sdr_local_model_1_residual,sdr_local_model_2_residual,sdr_local_model_3_residual,sdr_local_model_4_residual,sdr_local_model_5_residual,sdr_local_model_6_residual,sdr_local_model_7_residual,sdr_local_model_8_residual,sdr_IRM_residual, sdr_IRMVQ_residual];


isr_local_model_1_vocals = table2array(model_1_local_vocal(:,"isr"));
isr_local_model_2_vocals = table2array(model_2_local_vocal(:,"isr"));
isr_local_model_3_vocals = table2array(model_3_local_vocal(:,"isr"));
isr_local_model_4_vocals = table2array(model_4_local_vocal(:,"isr"));
isr_local_model_5_vocals = table2array(model_5_local_vocal(:,"isr"));
isr_local_model_6_vocals = table2array(model_6_local_vocal(:,"isr"));
isr_local_model_7_vocals = table2array(model_7_local_vocal(:,"isr"));
isr_local_model_8_vocals = table2array(model_8_local_vocal(:,"isr"));
isr_IRM_vocals = table2array(IRM_scores_vocals(:,"isr"));
isr_IRMVQ_vocals = table2array(IRMVQ_scores_vocals(:,"isr"));

ISR_vocals = [isr_local_model_1_vocals,isr_local_model_2_vocals,isr_local_model_3_vocals,isr_local_model_4_vocals,isr_local_model_5_vocals,isr_local_model_6_vocals,isr_local_model_7_vocals,isr_local_model_8_vocals,isr_IRM_vocals,isr_IRMVQ_vocals];

isr_local_model_1_residual = table2array(model_1_local_residual(:,"isr"));
isr_local_model_2_residual = table2array(model_2_local_residual(:,"isr"));
isr_local_model_3_residual = table2array(model_3_local_residual(:,"isr"));
isr_local_model_4_residual = table2array(model_4_local_residual(:,"isr"));
isr_local_model_5_residual = table2array(model_5_local_residual(:,"isr"));
isr_local_model_6_residual = table2array(model_6_local_residual(:,"isr"));
isr_local_model_7_residual = table2array(model_7_local_residual(:,"isr"));
isr_local_model_8_residual = table2array(model_8_local_residual(:,"isr"));
isr_IRM_residual = table2array(IRM_scores_residual(:,"isr"));
isr_IRMVQ_residual = table2array(IRMVQ_scores_residual(:,"isr"));
ISR_residual = [isr_local_model_1_residual,isr_local_model_2_residual,isr_local_model_3_residual,isr_local_model_4_residual,isr_local_model_5_residual,isr_local_model_6_residual,isr_local_model_7_residual,isr_local_model_8_residual,isr_IRM_residual,isr_IRMVQ_residual];


sar_local_model_1_vocals = table2array(model_1_local_vocal(:,"sar"));
sar_local_model_2_vocals = table2array(model_2_local_vocal(:,"sar"));
sar_local_model_3_vocals = table2array(model_3_local_vocal(:,"sar"));
sar_local_model_4_vocals = table2array(model_4_local_vocal(:,"sar"));
sar_local_model_5_vocals = table2array(model_5_local_vocal(:,"sar"));
sar_local_model_6_vocals = table2array(model_6_local_vocal(:,"sar"));
sar_local_model_7_vocals = table2array(model_7_local_vocal(:,"sar"));
sar_local_model_8_vocals = table2array(model_8_local_vocal(:,"sar"));
sar_noisy = table2array(noisy_scores(:,"sar"));
sar_IRM_vocals = table2array(IRM_scores_vocals(:,"sar"));
sar_IRMVQ_vocals = table2array(IRM_scores_vocals(:,"sar"));
SAR_vocals = [sar_local_model_1_vocals,sar_local_model_2_vocals,sar_local_model_3_vocals,sar_local_model_4_vocals,sar_local_model_5_vocals,sar_local_model_6_vocals,sar_local_model_7_vocals,sar_local_model_8_vocals,sar_IRM_vocals,sar_IRMVQ_vocals];


sar_local_model_1_residual = table2array(model_1_local_residual(:,"sar"));
sar_local_model_2_residual = table2array(model_2_local_residual(:,"sar"));
sar_local_model_3_residual = table2array(model_3_local_residual(:,"sar"));
sar_local_model_4_residual = table2array(model_4_local_residual(:,"sar"));
sar_local_model_5_residual = table2array(model_5_local_residual(:,"sar"));
sar_local_model_6_residual = table2array(model_6_local_residual(:,"sar"));
sar_local_model_7_residual = table2array(model_7_local_residual(:,"sar"));
sar_local_model_8_residual = table2array(model_8_local_residual(:,"sar"));
sar_IRM_residual = table2array(IRM_scores_residual(:,"sar"));
sar_IRMVQ_residual = table2array(IRMVQ_scores_residual(:,"sar"));

SAR_residual = [sar_local_model_1_residual, sar_local_model_2_residual, sar_local_model_3_residual, sar_local_model_4_residual, sar_local_model_5_residual, sar_local_model_6_residual, sar_local_model_7_residual, sar_local_model_8_residual, sar_IRM_residual, sar_IRMVQ_residual];

sir_local_model_1_vocals = table2array(model_1_local_vocal(:,"sir"));
sir_local_model_2_vocals = table2array(model_2_local_vocal(:,"sir"));
sir_local_model_3_vocals = table2array(model_3_local_vocal(:,"sir"));
sir_local_model_4_vocals = table2array(model_4_local_vocal(:,"sir"));
sir_local_model_5_vocals = table2array(model_5_local_vocal(:,"sir"));
sir_local_model_6_vocals = table2array(model_6_local_vocal(:,"sir"));
sir_local_model_7_vocals = table2array(model_7_local_vocal(:,"sir"));
sir_local_model_8_vocals = table2array(model_8_local_vocal(:,"sir"));
sir_noisy = table2array(noisy_scores(:,"sir"));
sir_IRM_vocals = table2array(IRM_scores_vocals(:,"sir"));
sir_IRMVQ_vocals = table2array(IRMVQ_scores_vocals(:,"sir"));

SIR_vocals = [sir_local_model_1_vocals,sir_local_model_2_vocals,sir_local_model_3_vocals,sir_local_model_4_vocals,sir_local_model_5_vocals,sir_local_model_6_vocals,sir_local_model_7_vocals,sir_local_model_8_vocals,sir_IRM_vocals,sir_IRMVQ_vocals];


sir_local_model_1_residual = table2array(model_1_local_residual(:,"sir"));
sir_local_model_2_residual = table2array(model_2_local_residual(:,"sir"));
sir_local_model_3_residual = table2array(model_3_local_residual(:,"sir"));
sir_local_model_4_residual = table2array(model_4_local_residual(:,"sir"));
sir_local_model_5_residual = table2array(model_5_local_residual(:,"sir"));
sir_local_model_6_residual = table2array(model_6_local_residual(:,"sir"));
sir_local_model_7_residual = table2array(model_7_local_residual(:,"sir"));
sir_local_model_8_residual = table2array(model_8_local_residual(:,"sir"));
sir_IRM_residual = table2array(IRM_scores_residual(:,"sir"));
sir_IRMVQ_residual = table2array(IRMVQ_scores_residual(:,"sir"));
SIR_residual = [sir_local_model_1_residual,sir_local_model_2_residual,sir_local_model_3_residual,sir_local_model_4_residual,sir_local_model_5_residual,sir_local_model_6_residual,sir_local_model_7_residual,sir_local_model_8_residual,sir_IRM_residual, sir_IRMVQ_residual];


xlabel={'MagSpec: w. rand. mix.','MagSpec: no rand. mix.','MagSpec: CCMSE loss','CompSpec: MSE loss','CompSpec: CCMSE loss','VQTSpec: MSE loss','MagSpecLog: MSE loss','MagSpec: no LR schedule','MagSpec: Ideal Ratio Mask', 'VQTSpec: Ideal Ratio Mask'};
window_size = [500 500 700 300];
figVec = [];
figCt=1;

rand_mix_id = strcmp(xlabel,'MagSpec: w. rand. mix.')|strcmp(xlabel,'MagSpec: no rand. mix.')|strcmp(xlabel,'MagSpec: no LR schedule');%|strcmp(xlabel,'MagSpec: Ideal Ratio Mask');

loss_comp_id = not(rand_mix_id);
loss_comp_id(end) = 1; % add Ideal-Ratio-Mask to plot
loss_comp_id(end-1) = 1; % add Ideal-Ratio-Mask to plot
loss_comp_id(1) = 1; % add basline to plot
loss_comp_labels = xlabel(loss_comp_id);
loss_comp_labels(1) = {'MagSpec: MSE loss'};
%%
sdr_vocal_vec = SDR_vocals(:,loss_comp_id);
sdr_vocal_vec = sdr_vocal_vec(:);

sar_vocal_vec = SAR_vocals(:,loss_comp_id);
sar_vocal_vec = sar_vocal_vec(:);

metric_vec = [sdr_vocal_vec; sar_vocal_vec];

row_names = cell(length(metric_vec),1);
row_names(1:length(metric_vec)/2,1) = {'SDR'};
row_names(length(metric_vec)/2+1:end,1) = {'SAR'};

loss_comp_label_mat = repmat(loss_comp_labels,50,1);
loss_comp_label_mat = loss_comp_label_mat(:);
loss_comp_label_mat = [loss_comp_label_mat;loss_comp_label_mat];

Metric_group = table(loss_comp_label_mat, metric_vec,row_names,'VariableNames',{'models','metrics','group_labels'});
Metric_group.models = categorical(Metric_group.models,loss_comp_labels);
Metric_group.group_labels = categorical(Metric_group.group_labels,{'SDR','SAR'});

figVec(figCt)=figure;
clf
hold on
grid on
grid minor
ylabel('ratio in dB')
boxchart(Metric_group.models, Metric_group.metrics,'GroupByColor',Metric_group.group_labels,'notch','on',MarkerStyle='x')%, BoxFaceColor=[86/255 112/255 147/255],MarkerColor=[86/255 112/255 147/255]);
xticklabels(loss_comp_labels)
ylim([-3 17])
yticks(-10:5:15)
set(gca,'FontSize',14)
xtickangle(labelangle)
set(gcf,'Position',window_size)
legend('Location','north')
figCt = figCt+1;


figVec(figCt)=figure;
clf
hold on
grid on
grid minor
ylabel('SDR in dB')
boxchart(ORACLE_SDR,'notch','on',MarkerStyle='x',BoxFaceColor=[86/255 112/255 147/255],MarkerColor=[86/255 112/255 147/255])
xticklabels({'noisy','MagSpec: Ideal Ratio Mask', 'VQTSpec: Ideal Ratio Mask'})
ylim([-15 15])     
yticks(-15:5:15)
set(gca,'FontSize',14)
title('\bf{dataset metrics: SDR}','FontSize',18)
xtickangle(labelangle)
figCt = figCt+1;

isr_vocal_vec = ISR_vocals(:,loss_comp_id);
isr_vocal_vec = isr_vocal_vec(:);

sir_vocal_vec = SIR_vocals(:,loss_comp_id);
sir_vocal_vec = sir_vocal_vec(:);

metric_vec = [isr_vocal_vec; sir_vocal_vec];

row_names = cell(length(metric_vec),1);
row_names(1:length(metric_vec)/2,1) = {'ISR'};
row_names(length(metric_vec)/2+1:end,1) = {'SIR'};

loss_comp_label_mat = repmat(loss_comp_labels,50,1);
loss_comp_label_mat = loss_comp_label_mat(:);
loss_comp_label_mat = [loss_comp_label_mat;loss_comp_label_mat];

Metric_group = table(loss_comp_label_mat, metric_vec,row_names,'VariableNames',{'models','metrics','group_labels'});
Metric_group.models = categorical(Metric_group.models,loss_comp_labels);
Metric_group.group_labels = categorical(Metric_group.group_labels,{'ISR','SIR'});


figVec(figCt)=figure;
clf
hold on
grid on
grid minor
ylabel('ratio in dB','FontSize',14)
boxchart(Metric_group.models, Metric_group.metrics,'GroupByColor',Metric_group.group_labels,'notch','on',MarkerStyle='x')%, BoxFaceColor=[86/255 112/255 147/255],MarkerColor=[86/255 112/255 147/255]);
ylim([-3 35])
yticks(-5:5:40)
xticklabels(loss_comp_labels)
set(gca,'FontSize',14)
ax= get(gca, 'XAxis');
xtickangle(labelangle)
set(gcf,'Position',window_size)
legend('Location','north')
figCt = figCt+1;

%% train loss vs. valid loss
clear xlabel
CompSpec_trainLoss = readtable('./evaluation_scores/run-CompSpec_MSE-tag-mean-training-loss per epoch_loss_.csv');
CompSpec_validLoss = readtable('./evaluation_scores/run-CompSpec_MSE-tag-mean-validation-loss per epoch_loss_.csv');
MagSpec_trainLoss = readtable('./evaluation_scores/run-MagSpec_MSE_w_LR_schedule-tag-mean-training-loss per epoch_loss_.csv');
MagSpec_validLoss = readtable('./evaluation_scores/run-MagSpec_MSE_w_LR_schedule-tag-mean-validation-loss per epoch_loss_.csv');
MagSpecLog_trainLoss = readtable('./evaluation_scores/run-MagSpecLog_MSE_w_LR_schedule-tag-mean-training-loss per epoch_loss_.csv');
MagSpecLog_validLoss = readtable('./evaluation_scores/run-MagSpecLog_MSE_w_LR_schedule-tag-mean-validation-loss per epoch_loss_.csv');

CompSpec_trainLoss = table2array([CompSpec_trainLoss(:,'Step'), CompSpec_trainLoss(:,'Value')]);
CompSpec_validLoss = table2array([CompSpec_validLoss(:,'Step'), CompSpec_validLoss(:,'Value')]);
MagSpec_trainLoss = table2array([MagSpec_trainLoss(:,'Step'), MagSpec_trainLoss(:,'Value')]);
MagSpec_validLoss = table2array([MagSpec_validLoss(:,'Step'), MagSpec_validLoss(:,'Value')]);
MagSpecLog_trainLoss = table2array([MagSpecLog_trainLoss(:,'Step'), MagSpecLog_trainLoss(:,'Value')]);
MagSpecLog_validLoss = table2array([MagSpecLog_validLoss(:,'Step'), MagSpecLog_validLoss(:,'Value')]);

diffCompSpec = (CompSpec_validLoss(end,2)-CompSpec_trainLoss(end,2));
diffMagSpec = (MagSpec_validLoss(end,2)-MagSpec_trainLoss(end,2));
diffMagSpecLog = (MagSpecLog_validLoss(end,2)-MagSpecLog_trainLoss(end,2));

figVec(figCt)=figure;
clf
hold on
grid on
colorOrder = get(gca, 'ColorOrder');
colorOrder(1,:) = [86/255 112/255 147/255];
p1=plot(CompSpec_trainLoss(:,1),CompSpec_trainLoss(:,2),'Linestyle','--','Color',colorOrder(1,:),'LineWidth',2);
p2=plot(CompSpec_validLoss(:,1),CompSpec_validLoss(:,2),'Linestyle','-','Color',colorOrder(1,:),'LineWidth',2);
p3=plot(MagSpec_trainLoss(:,1),MagSpec_trainLoss(:,2),'Linestyle','--','Color',colorOrder(2,:),'LineWidth',2);
p4=plot(MagSpec_validLoss(:,1),MagSpec_validLoss(:,2),'Linestyle','-','Color',colorOrder(2,:),'LineWidth',2);
p5=plot(MagSpecLog_trainLoss(:,1),MagSpecLog_trainLoss(:,2),'Linestyle','--','Color',colorOrder(3,:),'LineWidth',2);
p6=plot(MagSpecLog_validLoss(:,1),MagSpecLog_validLoss(:,2),'Linestyle','-','Color',colorOrder(3,:),'LineWidth',2);
set(gca,'FontSize',16)
ylabel('MSE loss ($\mathcal{L}_{MSE}$)')
xlabel('epochs')
ylim([0 4])
set(gcf,'Position',[500 500 700 350])
legend([p1,p2,p3,p4,p5,p6], 'CompSpec: MSE loss (training)','CompSpec: MSE loss (validation)', 'MagSpec: MSE loss (training)', 'MagSpec: MSE loss (validation)', 'MagSpecLog: MSE loss (training)', 'MagSpecLog: MSE loss (validation)','FontSize',16)
figCt = figCt+1;


%% save plots 
filenames = {'Vocals_SDR_SAR','dataset_metrics','Vocals_ISR_SIR','Loss_Comp_Train_vs_Valid'};

saveFigFlag = 1;
if saveFigFlag
    if(~exist('Figures', 'dir'))
        mkdir('Figures');
    end
    cd('Figures');
    for i=1:length(figVec)       
            set(figVec(i),'renderer','Painters')
            saveas(figVec(i), filenames{i},'epsc2');
    end
    cd('..');
end


