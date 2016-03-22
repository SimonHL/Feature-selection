% Extract exp data
% format:     Stress  Exp 27000 27000 27000
clear all;
clc;

data_all = load( 'data/data_all.mat');
load('data/data_detail_5457.mat'); % data
load('data/data_detail_9904.mat'); % data2

%% 
data_exp = zeros(8, 2 + 81000);

i = 1;

data_exp(i,1) = 45;
data_exp(i,2:end) = data(36,:); % 0045_5457
i = i + 1;

data_exp(i,1) = 45;
data_exp(i,2:end) = data2(70,:); % 0045_9904
i = i + 1;

data_exp(i,1) = 60;
data_exp(i,2:end) = data_all.data_006(168,:); % 006_1494
i = i + 1;

data_exp(i,1) = 80;
data_exp(i,2:end) = data_all.data_008(42,:); % 008_370
i = i + 1;

data_exp(i,1) = 90;
data_exp(i,2:end) = data_all.data_009(24,:); % 009_207
i = i + 1;

data_exp(i,1) = 100;
data_exp(i,2:end) = data_all.data_010(40,:); % 010_354
i = i + 1;

data_exp(i,1) = 130;
data_exp(i,2:end) = data_all.data_013(10,:); % 013_80
i = i + 1;

data_exp(i,1) = 130;
data_exp(i,2:end) = data_all.data_013(11,:); % 013_94
i = i + 1;

save('data/data_exp.mat', 'data_exp');







