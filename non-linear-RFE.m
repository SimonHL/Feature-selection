
% 非线性特征排序的主程序
% 用到的函数
% calculate_H


%% 数据加载和预处理  [train_ndata,train_nlabel]=preprocess;
% 
clear;
clc;
A=load('data/fatigue_life_for_vector_machine_zmh.dat');
train_label=A(:,end);
train_data=A(:,1:end-1);
train_data(:,2) = log10(train_data(:,2)); % 时间采用对数坐标

train_data=mapminmax(train_data')';  % 按列作规一化

% 删除y不同时，x相同的情况
pdata = train_data(train_label>0,:);
ndata = train_data(train_label<=0,:);
[overlap,ipdata,indata]= intersect(pdata,ndata,'rows');
pdata(ipdata,:)=[];
ndata(indata,:)=[];
p_size=size(pdata);
n_size=size(ndata);
train_label=[ones(p_size(1,1),1);-1*ones(n_size(1,1),1)];
train_data=[pdata;ndata];

save('data/fatigue_life_for_vector_machine_zmh.mat', 'train_data', 'train_label');

%% 用交叉验证的方式选择合适的参数
clear all;
clc;
load('data/fatigue_life_for_vector_machine_zmh.mat');
bestcv = 0;
for log2c = -10 : 10,
    for log2g = -5 : 5,
        cmd = ['-q -v 10 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(train_label, train_data, cmd);
        if (cv >= bestcv),
            bestcv = cv;
            bestc = 2^log2c;
            bestg = 2^log2g;
        end
    end
end
fprintf('log2c=%g log2g=%g cv=%g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);

save('data/fatigue_life_for_vector_machine_zmh.mat', 'train_data', 'train_label', 'bestc', 'bestg', 'bestcv');

%% 特征排序算法
clear all;
clc;
load('data/fatigue_life_for_vector_machine_zmh.mat');

[N_Sample, N_dim] = size(train_data);

% 提取alpha
cmd = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
alpha = calculate_alpha(train_data, train_label, cmd);

% 计算初始的代价函数矩阵
H = calculate_H(train_data, train_label, bestg);
A = train_data;
feature_index = 1:N_dim;
feature_removed = zeros(1,N_dim);

for t=N_dim:-1:2 % 只剩一个时不需要排序
    h=zeros(N_Sample,N_Sample,t);
    D=zeros(1,t);
    for i=1:t  % 剩于t个参数，需要计算去掉任意一个时的h矩阵
        A_temp=A;
        A_temp(:,i)=[];
%         A_temp(:,i) =  A_temp(randperm(N_Sample),i);
        h(:,:,i) = calculate_H(A_temp, train_label, bestg);
        alpha_h = calculate_alpha(A_temp, train_label, cmd);
        
        % 计算代价函数的变化
        D(i)=(1/2)*(alpha)'*H*alpha-(1/2)*(alpha_h)'*h(:,:,i)*alpha_h;  
    end
    
    [min_value, min_index] = min(D);

    feature_removed(t) = feature_index(min_index);
    feature_index(min_index) = [];
    A = train_data(:, feature_index); 
    H = h(:,:,i);
    alpha = calculate_alpha(A, train_label,cmd);
end
feature_removed(1) = feature_index; % 最后一个去掉的特征
