% Degree profile with Wasserstein distances
% A and B are the matrices to be matched 
% Return permutation matrix P so that P*A*P' is matched to B 

function [P_dp] = matching_deg_pro_wigner(A, B)
    n = size(A, 1);
    D = zeros(n);
    A_sort = sort(A); %sort columns
    B_sort = sort(B);

    for ind_i = 1:n
        for ind_j=1:n
            D(ind_i, ind_j) = sum(abs(A_sort(:, ind_i) - B_sort(:, ind_j))) / n;
            %Wasserstein distance
        end
    end
    
    M = matchpairs((-1)*D', -99999, 'max');
    P_dp = full(sparse(M(:, 1), M(:, 2), 1, n, n));
%     P_dp = full(greedy_match((-1)*D'));