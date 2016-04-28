function H = calculate_H_lin(x,y)
N_Sample = size(x,1);
H=zeros(N_Sample);
for i=1:N_Sample
    for j=1:N_Sample
        s=x(i,:)'*x(j,:);
        H(i,j)=y(i)*y(j)*s;
    end
end
end