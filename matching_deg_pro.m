% Degree profile with Wasserstein distances
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P_dp] = matching_deg_pro(A, B)
    n = size(A, 1);
    deg1=sum(A);       % degree sequence
    deg2=sum(B);

    [deg1_sort,ind1]=sort(deg1);
    [deg2_sort,ind2]=sort(deg2);


    DP1=cell(n,1);      % degree profile (sorted)        
    DP2=DP1;

    N_deg1=cell(n,1);
    F_deg1=cell(n,1);
    N_deg2=N_deg1;
    F_deg2=F_deg1;

    for i=1:1:n  
        temp1=deg1_sort(logical(A(i,ind1)));
        temp2=deg2_sort(logical(B(i,ind2))); 
        DP1{i}=temp1;
        DP2{i}=temp2;

        [temp1_unique,temp1_a,temp1_c]= unique(temp1);
        N_deg1{i}=temp1_unique;
        temp1_counts = accumarray(temp1_c,1);
        F_deg1{i}=temp1_counts/deg1(i);

        [temp2_unique,temp2_a,temp2_c]= unique(temp2);
        N_deg2{i}=temp2_unique;
        temp2_counts = accumarray(temp2_c,1);
        F_deg2{i}=temp2_counts/deg2(i); 
    end

    D=n*ones(n,n);
    for i=1:1:n
        if deg1(i)>0
            for j=1:1:n 
                if deg2(j)>0    
                D(i,j) =  dwass_discrete2(N_deg1{i},N_deg2{j},F_deg1{i},F_deg2{j});  
                %Wasserstein distance
                end
            end
        end
    end
    
    M = matchpairs((-1)*D', -99999, 'max');
    P_dp = full(sparse(M(:, 1), M(:, 2), 1, n, n));
%     P_dp = full(greedy_match((-1)*D'));