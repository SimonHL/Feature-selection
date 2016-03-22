function alpha = calculate_alpha( x,y, bestc, bestg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

model = svmtrain(y, x, ['-c ', num2str(bestc), ' -g ', num2str(bestg)]); % 使用优化后的参数得到model
alpha=zeros(size(y,1),1);
alpha(model.sv_indices)=model.sv_coef .* y(model.sv_indices); % alpha应为libsvm里sv_coef与y的乘积

end

