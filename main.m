%% 清理环境

clear all;
clc

% 设置常量
cell_number = 27000;

%%% 全程数据
file_name = 'data/data_for_zmh_0045.dat';
data_0045 = load_original_data(file_name,cell_number);

file_name = 'data/data_for_zmh_006.dat';
data_006 = load_original_data(file_name,cell_number);

file_name = 'data/data_for_zmh_008.dat';
data_008 = load_original_data(file_name,cell_number);

file_name = 'data/data_for_zmh_009.dat';
data_009 = load_original_data(file_name,cell_number);

file_name = 'data/data_for_zmh_010.dat';
data_010 = load_original_data(file_name,cell_number);

file_name = 'data/data_for_zmh_013.dat';
data_013 = load_original_data(file_name,cell_number);

save( 'data/data_all.mat')


%% 重新加载数据
clear all;
clc
load('data/data_all.mat');
load('data/data_exp.mat');


%% 分析标准方差变化趋势
[ std_t, std_t_D, std_t_DD, std_t_similar ] = std_analysis( data , cell_number);

%% 主特征分析

% 数据整理
data_all_cell = {data_0045, data_006, data_008, data_009, data_010, data_013};
data_all_cell_number = zeros(size(data_all_cell,2),1); 
for i = 1:length(data_all_cell_number)
    data_all_cell_number(i) = size(data_all_cell{i},1);
end

%  数据下标整理, 用以提取过程特征
data_index_begin = ones(size(data_all_cell_number));
data_index_end = data_all_cell_number;
for i = 2:length(data_all_cell_number);
    data_index_end(i) = sum(data_all_cell_number(1:i));
    data_index_begin(i) = data_index_end(i)  - data_all_cell_number(i) + 1;
end
data_index = [data_index_begin, data_index_end];

% 用所有的数据做主分量分析:该材料的分布特征， 从所有的仿真实验中提取
dim_t = sum(data_all_cell_number);
data = zeros(dim_t, cell_number*3+1);
for i = 1:length(data_all_cell_number)
    data(data_index(i,1):data_index(i,2),:) = data_all_cell{i};
end

%%
data_c = data(:,2:end);
coeff = zeros(cell_number,dim_t-1,3);
score = zeros(dim_t,dim_t-1,3);
mean_X = zeros(cell_number,3);
for i = 1:3  % 对三个物理量的分布特征分别分析
    data_X = data_c(:,(i-1)*cell_number+1:i*cell_number);
    [COEFF,SCORE,latent] = princomp(data_X,'econ');

    % 仿真过程中第i个物理量的分布特征和特征系数
    coeff(:,:,i) = COEFF;
    score(:,:,i) = SCORE;
    mean_X(:,i) = mean(data_X);
end

save('data/pca_data.mat', 'coeff', 'score', 'mean_X', 'data_all_cell','data_index', 'data', 'cell_number','data_all_cell_number', 'latent');


%% 提取出3个物理量分布第1主特征的演变过程
data_process = zeros(size(data,1),4); % 第一列是循环数，其他3列是3个物理量对主特征的系数
data_process(:,1) = data(:,1);
data_process(:,2) = score(:,1,1); % 第1个物理分量的第一个主特征
data_process(:,3) = score(:,1,2); % 第2个物理分量的第一个主特征
data_process(:,4) = score(:,1,3); % 第3个物理分量的第一个主特征

figure;   
charactor = 2; % 第1物理量的主特征变化

% 循环数和第一主特征的关系
plot(data_process(data_index(1,1):data_index(1,2),1), data_process(data_index(1,1):data_index(1,2),charactor), ...
     data_process(data_index(2,1):data_index(2,2),1), data_process(data_index(2,1):data_index(2,2),charactor), ...
     data_process(data_index(3,1):data_index(3,2),1), data_process(data_index(3,1):data_index(3,2),charactor), ...
     data_process(data_index(4,1):data_index(4,2),1), data_process(data_index(4,1):data_index(4,2),charactor), ...
     data_process(data_index(5,1):data_index(5,2),1), data_process(data_index(5,1):data_index(5,2),charactor), ...
     data_process(data_index(6,1):data_index(6,2),1), data_process(data_index(6,1):data_index(6,2),charactor) ); 
legend('0045', '006', '008', '009', '010', '013');
grid on; hold on;

% 计算寿命点对应的分布特征
data_c = data_exp(:,3:end);
exp_score = zeros(size(data_exp,1), size(coeff,2),3);
for i = 1:3  % 对三个物理量的分布特征分别分析
    data_X = data_c(:,(i-1)*cell_number+1:i*cell_number);
    exp_score(:,:,i) = (data_X-repmat(mean_X(:,i)', size(data_X,1),1)) * coeff(:,:,i);  % 计算变换后的特征
end

% 寿命点对应的主特征
for i = 1:size(exp_score,1)
    plot(data_exp(i,2), exp_score(i,1,charactor-1) ,'*');
end

%%
clear all;
clc;
load('data/pca_data.mat');
load('data/data_exp.mat');


%% 线性分量回归分析

close all;

% 从不同的数据中提取出标准差特征
feature_lin_var =  zeros(length(data_all_cell_number),3); 
for i = 1:length(data_all_cell_number)    
    [ std_t, std_t_D, std_t_DD, std_t_similar ] = std_analysis( data_all_cell{i} , cell_number);
    feature_lin_var(i,:) = 1./mean(std_t_D(:,2:end)./repmat(std_t_D(:,1),[1 3]), 1);
end

% 分布第1主分量特征
feature_lin_lambda = zeros(length(data_all_cell_number),3); 
for i = 1:length(data_all_cell_number)
    for j = 1:3 % 3个物理特征
        tmp_score = score( data_index(i,1): data_index(i,2) ,1,j);  % 第j个物理量的第1个主特征
        
        tmp_time_D = diff(data_all_cell{i}(:,1),1);% 数据的时间点并不是均匀的
        feature_lin_lambda(i,j) = 1./mean( diff(tmp_score,1)./ tmp_time_D);
    end
end

% 构造进行线性回归的X和Y
sample_num = 8;
x_input = zeros(6,sample_num);
y_input = zeros(1,sample_num);
for i = 1:sample_num
    
    % 重复实验处理
    if i == 1
        tmp_index = 1;
    elseif i == sample_num
        tmp_index = 6;
    else
        tmp_index = i-1;
    end
    
    x_input(1:3,i) = feature_lin_var(tmp_index,:);
    x_input(4:end,i) =  feature_lin_lambda(tmp_index,:);
    y_input(1,i) = data_exp(i,2);
end


% [y,ps_y] = mapminmax(y_input, min(y_input),max(y_input));
[x,ps_x] = mapminmax(x_input, -1,1);

% 选择参数C
best_cor = 0;
bestc = 1000;
for log2c = -10 : 30,
        cmd = ['-q -s 3 -t 0 -p 0.01 -c ', num2str(2^log2c)];
        model = svmtrain(y', x', cmd);
        [y_pred, accuracy, prob_estimates] = svmpredict( y', x', model, ' -q');
        cor = accuracy(3);
        if (cor > best_cor),
            best_cor = cor;
            bestc = 2^log2c;
        end
end

str_option = ['-s 3 -t 0 -p 0.01 -c ', num2str(bestc)];
model = svmtrain(y', x', str_option); 

w = model.SVs' * model.sv_coef;
w2 = w.^2;

% y_pred = x'* w - model.rho;

% accuracy是一个3*1的列向量，其中第1个数字用于分类问题，表示分类准确率；后两个数字用于回归问题，第2个数字表示mse；第三个数字
% 表示平方相关系数（也就是说，如果分类的话，看第一个数字就可以了；回归的话，看后两个数字）。
[y_pred, accuracy, prob_estimates] = svmpredict( y', x', model, ' -q');    % 测试时svmpredict的y没有关系，目前只需要得到预测值y_pred
y_pred = mapminmax('reverse', y_pred', ps_y);
view_point = 5;
plot(x_input(view_point,:),y_input,'bo-', x_input(view_point,:),y_pred,'rs-');
figure;
plot(x(view_point,:),y_input,'bo-', x(view_point,:),y_pred,'rs-');

%%
[ std_t, std_t_D, std_t_DD, std_t_similar ] = std_analysis( data_all_cell{1} , cell_number);












