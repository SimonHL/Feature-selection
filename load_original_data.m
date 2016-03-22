function [ data ] = load_original_data( file_name, cell_number )
% ��ԭʼ�����ж�ȡ���������������Ϊ����������
%  data�Ľṹ��   ������  ����1���ݣ��ӵ�Ԫ1����ԪN��   ����2����   ����3����

data_origin = load(file_name, '-ascii');
t_span = unique(data_origin(:,1));
dim_t = length(t_span);
data_tmp = data_origin(:,[3 4 5]); % ��������Ҫ�ϲ��Ĳ���
data = zeros(dim_t, 1 + cell_number*3); % ������  + 3��ԭʼ��������

for i = 1:dim_t
    begin_index = (i-1) * cell_number;
    data_block = data_tmp(begin_index+1:begin_index+cell_number,:);
    data(i,1)=t_span(i);
    data(i,2:end) = data_block(:);% ������ϣ���ǰ1/3������1����2/3������3
end

end

