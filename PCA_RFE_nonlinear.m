clear all;
clc;
load('data/pca_data.mat');
load('data/data_exp.mat');


%% 提取特征
clc;
% data的第一列是循环数，其余为3个物理量对应的特征系数，共27000*3=81000个
% data_index 对应不同的6次实验


% 计算寿命点对应的分布特征
data_c = data_exp(:,3:end);
exp_score = zeros(size(data_exp,1), size(coeff,2),3);
for i = 1:3  % 对三个物理量的分布特征分别分析
    data_X = data_c(:,(i-1)*cell_number+1:i*cell_number);
    exp_score(:,:,i) = (data_X-repmat(mean_X(:,i)', size(data_X,1),1)) * coeff(:,:,i);  % 计算变换后的特征
end

% 构造进行线性回归的X和Y
sample_num = 8;
end_charactor = 30;
x_input = [exp_score(:,1:end_charactor,1) exp_score(:,1:end_charactor,2) exp_score(:,1:end_charactor,3)]';
y_input = zeros(1,sample_num);
y_input(1,:) = data_exp(:,2);

y = y_input;
[y,ps_y] = mapminmax(y_input, min(y_input),max(y_input));
[x,ps_x] = mapminmax(x_input, -1,1);


% 选择参数
cmd_share = ' -q -s 3 -p 0.001 -c ';
best_result = Inf;
log2c_range = -10 : 30;
log2g_range = -5:5;
bestc = 0;
bestg = 0;
choose_mode = 2;   % 1： 线性  2 非线性
for log2c = log2c_range
    if choose_mode == 1  % 线性
        cmd = ['-t 0 -v 8 ',cmd_share , num2str(2^log2c)]; % 留1验证
        cv_result = svmtrain(y', x', cmd);
        if (cv_result < best_result),
            best_result = cv_result;
            bestc = 2^log2c;
        end
    else   % 非线性
        for log2g = log2g_range
            cmd = ['-t 2 -v 8 ',cmd_share , num2str(2^log2c), ' -g ', num2str(2^log2g)]; % 留1验证
            cv_result = svmtrain(y', x', cmd);
            if (cv_result < best_result),
                best_result = cv_result;
                bestc = 2^log2c;
                bestg = 2^log2g;
            end
        end
    end
end

if choose_mode == 1  % 线性
    str_option = [' -t 0  ', cmd_share , num2str(bestc)];
else
    str_option = [' -t 2  ', cmd_share , num2str(bestc), ' -g ', num2str(bestg)];
end


model = svmtrain(y', x', str_option); 
w = model.SVs' * model.sv_coef;
w2 = w.^2;

% y_pred = x'* w - model.rho;
% accuracy是一个3*1的列向量，其中第1个数字用于分类问题，表示分类准确率；后两个数字用于回归问题，第2个数字表示mse；第三个数字
% 表示平方相关系数（也就是说，如果分类的话，看第一个数字就可以了；回归的话，看后两个数字）。
[y_pred, accuracy, prob_estimates] = svmpredict( y', x', model, ' -q');    % 测试时svmpredict的y没有关系，目前只需要得到预测值y_pred
y_pred = mapminmax('reverse', y_pred', ps_y);
view_point = 5;
% figure;
% plot(x(view_point,:),y_input,'bo-', x(view_point,:),y_pred,'rs-');

% RFE

train_data = x';
train_label = y';

[N_Sample, N_dim] = size(train_data);

% 提取alpha
alpha = calculate_alpha(train_data, train_label,str_option);

% 计算初始的代价函数矩阵
H = calculate_H(train_data, train_label, bestg, choose_mode);
A = train_data;
feature_index = 1:N_dim;
feature_removed = zeros(1,N_dim);

for t=N_dim:-1:2 % 只剩一个时不需要排序
    
    h=zeros(N_Sample,N_Sample,t);
    D=zeros(1,t);
    for i=1:t  % 剩于t个参数，需要计算去掉任意一个时的h矩阵
        A_temp=A;
        A_temp(:,i)=mean(A_temp(:,i));
%         A_temp(:,i) =  A_temp(randperm(N_Sample),i);
        h(:,:,i) = calculate_H(A_temp, train_label, bestg, choose_mode);
%         alpha_h = calculate_alpha(A_temp, train_label, str_option);
        
        % 计算代价函数的变化
        D(i)=(1/2)*(alpha)'*H*alpha-(1/2)*(alpha)'*h(:,:,i)*alpha;  
%         D(i)=(1/2)*(alpha)'*H*alpha-(1/2)*(alpha_h)'*h(:,:,i)*alpha_h + sum(alpha) - sum(alpha_h);  
    end
    
    [min_value, min_index] = min(D);
    [max_value, max_index] = max(D);
%     [D(1) D(2) min_value max_value]
    fprintf('%.0f : %.0f  (%.0f)  \n',t, feature_index(min_index),  mod(feature_index(min_index), end_charactor ) );
    feature_removed(t) = feature_index(min_index);
    feature_index(min_index) = [];
    A = train_data(:, feature_index); 
    H = h(:,:,i);
    alpha = calculate_alpha(A, train_label, str_option);
end
feature_removed(1) = feature_index; % 最后一个去掉的特征
feature_index_mod = mod(feature_removed, end_charactor);
feature_index_mod(feature_index_mod == 0) = end_charactor;
 fprintf('%.0f : %.0f  (%.0f) \n',1,  feature_index, mod(feature_index, end_charactor ) );
 fprintf('End \n ');
 
 %% 排序结果验证
 if choose_mode == 1
     test_cmd = ['-t 0 -v 8 ',cmd_share , num2str(bestc)]; % 留1验证
 else
     test_cmd = ['-t 2 -v 8 ',cmd_share , num2str(bestc), ' -g ', num2str(bestg)]; % 留1验证
 end
 test_cv = zeros(1,N_dim);
 for i = 1:N_dim
     test_data = train_data(:,feature_removed(1:N_dim-i+1));
%      test_data = train_data(:,1:N_dim-i+1);
     test_cv(i) = svmtrain(train_label,test_data, test_cmd);    
 end
 figure;
 plot(test_cv);



