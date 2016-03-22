function [ std_t, std_t_D, std_t_DD, std_t_similar ] = std_analysis( data , cell_number)
%   从数据中计算对应时刻的标准差
%
dim_t = size(data,1);
data_c = data(:,2:end); % 只含数据的部分
std_t = zeros(dim_t,4);
for i = 1:dim_t
    std_t(i,1) = data(i,1);  % 周期数
    for j =1:3 % 三个原始特征
        std_t(i,j+1)= std(data_c(i, (j-1)*cell_number+1 : j*cell_number));
    end
end

std_t_D = diff(std_t,1);
std_t_DD =  diff(std_t,2);  % 二阶差分平稳
std_t_similar = std_t_DD(:,2).* std_t_DD(:,3).*std_t_DD(:,4);

end

