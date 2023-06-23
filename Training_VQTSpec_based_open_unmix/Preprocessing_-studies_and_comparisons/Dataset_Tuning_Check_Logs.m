


set(0,'defaulttextInterpreter','latex')
set(0,'defaultlegendInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex')

fs = 44100;

test_set_logs = open('./tuning_logs/test_set_tuning_log.mat').tuning_log;

train_set_logs = open('./tuning_logs/train_set_tuning_log.mat').tuning_log;

test_set_fTune = cell2mat({test_set_logs{1,:,1}});
test_set_detuneFact = cell2mat({test_set_logs{1,:,2}});
test_set_prevLength = cell2mat({test_set_logs{1,:,3}});
test_set_postLength = cell2mat({test_set_logs{1,:,4}});

test_set_control_detuneFact = test_set_postLength./test_set_prevLength;





train_set_fTune = cell2mat({train_set_logs{1,:,1}});
train_set_detuneFact = cell2mat({train_set_logs{1,:,2}});
train_set_prevLength = cell2mat({train_set_logs{1,:,3}});
train_set_postLength = cell2mat({train_set_logs{1,:,4}});

train_set_control_detuneFact = train_set_postLength./train_set_prevLength;



figure
subplot(3,1,1)
hold on
grid on
%plot(1:length(train_set_control_detuneFact),train_set_control_detuneFact)
plot(1:length(train_set_control_detuneFact),train_set_detuneFact)
xlim([1 length(train_set_control_detuneFact)])
xlabel('sample id')
ylabel('detuning-factor')
title('detuning-factor')
subplot(3,1,2)
hold on
grid on
plot(1:length(train_set_fTune),train_set_fTune)
plot(1:length(train_set_fTune),train_set_fTune./train_set_control_detuneFact)
xlim([1 length(train_set_fTune)])
legend('tuning frequency before detuning','tuning frequency after detuning')
ylabel('tuning frequency in Hz')
xlabel('sample id')
title('tuning-frequency')
subplot(3,1,3)
hold on
grid on
plot(1:length(train_set_prevLength), (train_set_prevLength-train_set_postLength)/fs)
%plot(1:length(train_set_postLength), train_set_postLength)
xlabel('sample id')
ylabel('sample-length devation in seconds')
legend('sample-length deviation')
title('sample-length deviation')


figure
subplot(3,1,1)
hold on
grid on
%plot(1:length(train_set_control_detuneFact),train_set_control_detuneFact)
plot(1:length(test_set_control_detuneFact),test_set_detuneFact)
xlim([1 length(test_set_control_detuneFact)])
xlabel('sample id')
ylabel('detuning-factor')
title('detuning-factor')
subplot(3,1,2)
hold on
grid on
plot(1:length(test_set_fTune),test_set_fTune)
plot(1:length(test_set_fTune),test_set_fTune./test_set_control_detuneFact)
xlim([1 length(test_set_fTune)])
legend('tuning frequency before detuning','tuning frequency after detuning')
ylabel('tuning frequency in Hz')
xlabel('sample id')
title('tuning-frequency')
subplot(3,1,3)
hold on
grid on
plot(1:length(test_set_prevLength), (test_set_prevLength-test_set_postLength)/fs)
%plot(1:length(train_set_postLength), train_set_postLength)
xlabel('sample id')
% xticks([1:4])
% xticklabels(['$\Delta$','$\Delta$'])
ylabel('sample-length devation in seconds')
legend('sample-length deviation')
title('sample-length deviation')

%legend('sample-length before detuning','sample-length after detuning')



