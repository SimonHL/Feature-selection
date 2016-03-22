function svr_test
%  SVR 功能测试
%  分别测试SVR对不同类型函数的拟合能力

%% 数据生成
clear all;
clc
x_min = 0.1;
x_max = 0.9;
x_delta = (x_max-x_min)/10;  % 根据样本数来定步长
x =      x_min: x_delta: x_max;
x_pred = x_min : x_delta: x_max + 1.5*(x_max-x_min);

%% SVR 测试

close all;
str_option = '-s 3 -t 0 -p 0.001';
func_tmp = @func_lin;
y = func_tmp(x);
y_pred = regression_x_y(x,y,str_option,x_pred);
figure;
plot(x_pred, func_tmp(x_pred), 'b*-', x_pred, y_pred, 'ro-');legend('真实值','回归值');grid on;
title('线性回归结果');

str_option = '-s 3 -t 2 -g 1 -c 1000 -p 0.00001 ';
func_tmp = @func_exp_neg; 
y = func_tmp(x);
y_pred = regression_x_y(1./x,y,str_option,1./x_pred); 
figure;
plot(x_pred, func_tmp(x_pred), 'b*-', x_pred, y_pred, 'ro-');legend('真实值','回归值');grid on;
title('非线性回归  - 负指数关系');

str_option = '-s 3 -t 2 -g 1 -c 1000  -p 0.00001 ';
func_tmp = @func_exp_positive; 
y = func_tmp(x);
y_pred = regression_x_y(x,y,str_option,x_pred);
figure;
plot(x_pred, func_tmp(x_pred), 'b*-', x_pred, y_pred, 'ro-');legend('真实值','回归值');grid on;
title('非线性回归  - 正指数关系');


str_option = '-s 3 -t 2 -g 1 -c 1000  -p 0.00001 ';
func_tmp = @func_sin; 
y = func_tmp(x);
y_pred = regression_x_y(x,y,str_option,x_pred);
figure;
plot(x_pred, func_tmp(x_pred), 'b*-', x_pred, y_pred, 'ro-');legend('真实值','回归值');grid on;
title('非线性回归  - 正弦关系');


end

%%% 目标函数的定义
function [y_pred] = regression_x_y(x,y,str_option, x_pred)
    [y,ps] = mapminmax(y, min(x),max(x));
    model = svmtrain(y', x', str_option); 
    y_pred = svmpredict( x_pred', x_pred', model, ' -q');    % 测试时svmpredict的y没有关系，目前只需要得到预测值y_pred
    y_pred = mapminmax('reverse', y_pred', ps);
end

function y = func_lin(x)
    y = 3 * x + 11 + 0.0*randn(size(x));      % 线性回归测试
end

function y = func_exp_neg(x)
    y  = 0.1 * x.^(-1.5) + 35 + 0.0*randn(size(x));    % 非线性回归  - 负指数关系
end

function y = func_exp_positive(x)
    y = 0.65 * x.^0.5 + 7.12;      % 非线性回归  - 正指数关系
end

function y = func_sin(x)
 y = 0.38 * sin(x-0.2) + 0.56;     % 非线性回归  - 正弦关系 
end


