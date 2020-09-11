%% compute schur complement

function [S] = shur_complement(V_A, V_B, lambda,n)

S1=zeros(n,n);
V_B_one=V_B'*ones(n,1);
V_A_one=V_A'*ones(n,1);
for i=1:1:n
    temp=lambda((i-1)*n+1:i*n)*ones(1,n);
    S1=S1+V_B_one(i)^2*V_A*(temp.*V_A');
end


S2=zeros(n,n);
for i=1:1:n
    temp=lambda((i-1)*n+1:i*n).*V_A_one;
    S2=S2+V_B_one(i)*V_A*(temp*V_B(:,i)');
end

S3=S2';

S4=zeros(n,n);
for i=1:1:n
    temp=lambda((i-1)*n+1:i*n).*(V_A_one.^2);
    S4=S4+sum(temp)*V_B(:,i)*V_B(:,i)';
end

S=(-1)*[S1,S2;S3,S4];



