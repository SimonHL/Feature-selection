clear all;
clc;
load('data/pca_data.mat');
load('data/data_exp.mat');


%% ��ȡ����
clc;
% data�ĵ�һ����ѭ����������Ϊ3����������Ӧ������ϵ������27000*3=81000��
% data_index ��Ӧ��ͬ��6��ʵ��


% �����������Ӧ�ķֲ�����
data_c = data_exp(:,3:end);
exp_score = zeros(size(data_exp,1), size(coeff,2),3);
for i = 1:3  % �������������ķֲ������ֱ����
    data_X = data_c(:,(i-1)*cell_number+1:i*cell_number);
    exp_score(:,:,i) = (data_X-repmat(mean_X(:,i)', size(data_X,1),1)) * coeff(:,:,i);  % ����任�������
end

% ����������Իع��X��Y
sample_num = 8;
end_charactor = 30;
x_input = [exp_score(:,1:end_charactor,1) exp_score(:,1:end_charactor,2) exp_score(:,1:end_charactor,3)]';
y_input = zeros(1,sample_num);
y_input(1,:) = data_exp(:,2);

y = y_input;
[y,ps_y] = mapminmax(y_input, min(y_input),max(y_input));
[x,ps_x] = mapminmax(x_input, -1,1);


% ѡ�����
cmd_share = ' -q -s 3 -p 0.001 -c ';
best_result = Inf;
log2c_range = -10 : 30;
log2g_range = -5:5;
bestc = 0;
bestg = 0;
choose_mode = 2;   % 1�� ����  2 ������
for log2c = log2c_range
    if choose_mode == 1  % ����
        cmd = ['-t 0 -v 8 ',cmd_share , num2str(2^log2c)]; % ��1��֤
        cv_result = svmtrain(y', x', cmd);
        if (cv_result < best_result),
            best_result = cv_result;
            bestc = 2^log2c;
        end
    else   % ������
        for log2g = log2g_range
            cmd = ['-t 2 -v 8 ',cmd_share , num2str(2^log2c), ' -g ', num2str(2^log2g)]; % ��1��֤
            cv_result = svmtrain(y', x', cmd);
            if (cv_result < best_result),
                best_result = cv_result;
                bestc = 2^log2c;
                bestg = 2^log2g;
            end
        end
    end
end

if choose_mode == 1  % ����
    str_option = [' -t 0  ', cmd_share , num2str(bestc)];
else
    str_option = [' -t 2  ', cmd_share , num2str(bestc), ' -g ', num2str(bestg)];
end


model = svmtrain(y', x', str_option); 
w = model.SVs' * model.sv_coef;
w2 = w.^2;

% y_pred = x'* w - model.rho;
% accuracy��һ��3*1�������������е�1���������ڷ������⣬��ʾ����׼ȷ�ʣ��������������ڻع����⣬��2�����ֱ�ʾmse������������
% ��ʾƽ�����ϵ����Ҳ����˵���������Ļ�������һ�����־Ϳ����ˣ��ع�Ļ��������������֣���
[y_pred, accuracy, prob_estimates] = svmpredict( y', x', model, ' -q');    % ����ʱsvmpredict��yû�й�ϵ��Ŀǰֻ��Ҫ�õ�Ԥ��ֵy_pred
y_pred = mapminmax('reverse', y_pred', ps_y);
view_point = 5;
% figure;
% plot(x(view_point,:),y_input,'bo-', x(view_point,:),y_pred,'rs-');

% RFE

train_data = x';
train_label = y';

[N_Sample, N_dim] = size(train_data);

% ��ȡalpha
alpha = calculate_alpha(train_data, train_label,str_option);

% �����ʼ�Ĵ��ۺ�������
H = calculate_H(train_data, train_label, bestg, choose_mode);
A = train_data;
feature_index = 1:N_dim;
feature_removed = zeros(1,N_dim);

for t=N_dim:-1:2 % ֻʣһ��ʱ����Ҫ����
    
    h=zeros(N_Sample,N_Sample,t);
    D=zeros(1,t);
    for i=1:t  % ʣ��t����������Ҫ����ȥ������һ��ʱ��h����
        A_temp=A;
        A_temp(:,i)=mean(A_temp(:,i));
%         A_temp(:,i) =  A_temp(randperm(N_Sample),i);
        h(:,:,i) = calculate_H(A_temp, train_label, bestg, choose_mode);
%         alpha_h = calculate_alpha(A_temp, train_label, str_option);
        
        % ������ۺ����ı仯
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
feature_removed(1) = feature_index; % ���һ��ȥ��������
feature_index_mod = mod(feature_removed, end_charactor);
feature_index_mod(feature_index_mod == 0) = end_charactor;
 fprintf('%.0f : %.0f  (%.0f) \n',1,  feature_index, mod(feature_index, end_charactor ) );
 fprintf('End \n ');
 
 %% ��������֤
 if choose_mode == 1
     test_cmd = ['-t 0 -v 8 ',cmd_share , num2str(bestc)]; % ��1��֤
 else
     test_cmd = ['-t 2 -v 8 ',cmd_share , num2str(bestc), ' -g ', num2str(bestg)]; % ��1��֤
 end
 test_cv = zeros(1,N_dim);
 for i = 1:N_dim
     test_data = train_data(:,feature_removed(1:N_dim-i+1));
%      test_data = train_data(:,1:N_dim-i+1);
     test_cv(i) = svmtrain(train_label,test_data, test_cmd);    
 end
 figure;
 plot(test_cv);



