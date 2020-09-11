% spectral initialization method
function [S0] = spectral_init(W1, W2)

    % align leading eigenvectors of W1 and W2
             k=1;
             [V1,D1]=eigs(W1,k,'la');
             [V2,D2]=eigs(W2,k,'la');
             temp=V2*V1';
             
             [S1,] = greedy_match(temp);
             [S2,] = greedy_match((-1)*temp);
              val1=sum(dot(S1,temp));
              val2=(-1)*sum(dot(S2,temp));
             % [val1,m_i,m_j]=bipartite_matching(temp);
             % [val2,n_i,n_j]=bipartite_matching((-1)*temp);
              
             %   n=size(W1,1);
             %   S0=zeros(n,n);
             
            if val1>val2
                S0=S1;
%               S0(m_i,m_j)=eye(n);
            else
                S0=S2;
%                S0(n_i,n_j)=eye(n);
               
             end