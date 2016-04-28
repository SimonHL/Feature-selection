function alpha = calculate_alpha( x,y, str_option)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
model = svmtrain(y, x, str_option); % 使用优化后的参数得到model
alpha=zeros(size(y,1),1);
alpha(model.sv_indices)=model.sv_coef .* y(model.sv_indices); % alpha应为libsvm里sv_coef与y的乘积

end

