function alpha = calculate_alpha( x,y, bestc, bestg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

model = svmtrain(y, x, ['-c ', num2str(bestc), ' -g ', num2str(bestg)]); % ʹ���Ż���Ĳ����õ�model
alpha=zeros(size(y,1),1);
alpha(model.sv_indices)=model.sv_coef .* y(model.sv_indices); % alphaӦΪlibsvm��sv_coef��y�ĳ˻�

end

