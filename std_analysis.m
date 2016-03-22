function [ std_t, std_t_D, std_t_DD, std_t_similar ] = std_analysis( data , cell_number)
%   �������м����Ӧʱ�̵ı�׼��
%
dim_t = size(data,1);
data_c = data(:,2:end); % ֻ�����ݵĲ���
std_t = zeros(dim_t,4);
for i = 1:dim_t
    std_t(i,1) = data(i,1);  % ������
    for j =1:3 % ����ԭʼ����
        std_t(i,j+1)= std(data_c(i, (j-1)*cell_number+1 : j*cell_number));
    end
end

std_t_D = diff(std_t,1);
std_t_DD =  diff(std_t,2);  % ���ײ��ƽ��
std_t_similar = std_t_DD(:,2).* std_t_DD(:,3).*std_t_DD(:,4);

end

