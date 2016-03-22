function H = calculate_H(x,y,gamma)
N_Sample = size(x,1);
H=zeros(N_Sample);
for i=1:N_Sample
    for j=1:N_Sample
        s=sum((x(i,:)-x(j,:)).^2);
        K=exp(-s*gamma);
        H(i,j)=y(i)*y(j)*K;
    end
end
end