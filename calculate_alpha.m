function alpha = calculate_alpha( x,y, str_option)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
model = svmtrain(y, x, str_option); % ʹ���Ż���Ĳ����õ�model
alpha=zeros(size(y,1),1);
alpha(model.sv_indices)=model.sv_coef .* y(model.sv_indices); % alphaӦΪlibsvm��sv_coef��y�ĳ˻�

end

