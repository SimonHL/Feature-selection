function [ data ] = load_original_data( file_name, cell_number )
% 从原始数据中读取并整理出以周期数为样本的数据
%  data的结构：   周期数  特征1数据（从单元1到单元N）   特征2数据   特征3数据

data_origin = load(file_name, '-ascii');
t_span = unique(data_origin(:,1));
dim_t = length(t_span);
data_tmp = data_origin(:,[3 4 5]); % 数据中需要合并的部分
data = zeros(dim_t, 1 + cell_number*3); % 周期数  + 3种原始特征数据

for i = 1:dim_t
    begin_index = (i-1) * cell_number;
    data_block = data_tmp(begin_index+1:begin_index+cell_number,:);
    data(i,1)=t_span(i);
    data(i,2:end) = data_block(:);% 按列组合，及前1/3是特征1，后2/3是特征3
end

end

