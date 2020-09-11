function [z, history] = quadprog_admm(A,B,Aeq,b,lb, ub, rho, alpha,ABSTOL,RELTOL)
% quadprog  Solve standard form box-constrained QP via ADMM
%
% [x, history] = quadprog(P, q, r, lb, ub, rho, alpha)
%
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*x'*P*x + q'*x + r
%   subject to   Aeq *x = b;
%                 lb <= x <= ub
%  where P= kron(speye(n),A'*A)+kron(B*B',speye(n))-2*kron(B',A');
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

%Global constants and defaults

QUIET    =0;
%MAX_ITER = 1000;
MAX_ITER = 100;
ABSTOL   = 1e-5;
RELTOL   = 1e-3;

%Data preprocessing

n = size(A,1);

%ADMM solver

x = zeros(n^2,1);
z = zeros(n^2,1);
u = zeros(n^2,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective','rho');
end


% preprocessing: 

        %compute (P+ rho* I )^(-1)      
        [V_A, D_A]=eig(A);
        [V_B, D_B]=eig(B);
        d_A=diag(D_A);
        d_B=diag(D_B);
        
        Lambda=d_A.^2*ones(1,n)+ones(n,1)*(d_B').^2-2*d_A*d_B' +rho*ones(n,n);        
        Lambda_inv=1./Lambda;
        lambda=Lambda_inv(:);
        
        %Lambda_kron= diag(lambda);
        
        
       % V_BA=[kron(ones(1,n)*V_B,V_A);kron(V_B,ones(1,n)*V_A)];     %2n*n^2
        % I guess this step is the bottlenecker where we used n^2*n^2
        % matrix; Matlab runs out of memory in this step. I do not know how to simply this step.
       
        
       % S=(-1)*V_BA*((lambda*ones(1,2*n)).*V_BA');  %schur complement
  %  tic;
       S=shur_complement(V_A,V_B,lambda,n);
  % toc;
    
% compute the basis vectors perpendicular to [1_n, -1_n]
        % this step can be simplified by constructing Fourier basis
   %     temp=full(Aeq*Aeq');
   %     [V_temp,D_temp]=eigs(temp,2*n-1,'LA');    % the columns of V_temp are perpendicular to [1_n, -1_n];
   v_f=Fourier_basis(n);
   V_temp1=[v_f(:,1)/sqrt(2),v_f(:,2:n),zeros(n,n-1)];
   V_temp2=[v_f(:,1)/sqrt(2),zeros(n,n-1),v_f(:,2:n)];
   V_temp=[V_temp1;V_temp2];


    S_proj=V_temp'*S*V_temp;
%        Aeq_proj=V_temp'*Aeq;
        b_proj=V_temp'*b;
%         KKT_matrix=[P+rho*eye(n^2), Aeq'*V_temp;V_temp'*Aeq,zeros(2*n-1,2*n-1)];
%         KKT_matrix_inv=KKT_matrix^(-1);    % this takes too much time to
%         inverse
for k = 1:MAX_ITER

   
    
   
      %  x = R \ (R' \ (rho*(z - u) - q));
        
        
        
   
        %R = chol(P + rho*eye(n) );
        %x = R \ (R' \ (rho*(z - u) - q));
        % first iteration
        
        
        
        % compute (H+rho*I)^(-1) (z-u) =vec(mat_1);
     
        Z=vec2mat(z,n)';
        U=vec2mat(u,n)';
        vec_1=rho*V_A'*(Z-U)*V_B;
        vec_2=lambda.*vec_1(:);
        mat_1=V_A*vec2mat(vec_2,n)'*V_B';
        
        % compute -A'mu
       
        Aeq_mat1=[mat_1*ones(n,1);mat_1'*ones(n,1)];
        mu=linsolve(S_proj, V_temp'*Aeq_mat1 - b_proj);
        V_temp_mu=V_temp*mu;
        %vec3=Aeq'*(V_temp*mu);
        vec3=kron(ones(n,1),V_temp_mu(1:n))+kron(V_temp_mu(n+1:2*n),ones(n,1));
        % compute - (H+I)^-1 (A' mu)
        vec_4=V_A'*vec2mat(vec3,n)'*V_B;
        vec_5=lambda.*vec_4(:);
        mat_2=V_A*vec2mat(vec_5,n)'*V_B';
        
        x=mat_2+mat_1;
        
        x=x(:);     
%  % alternatively, we can construct the inverse of KKT matrix
%
% tic;
%  x_temp=KKT_matrix_inv*[z-u;V_temp'*b];    
%  x=x_temp(1:n^2);
% toc;
% mu=(-1)*x_temp(n^2+1:n^2+2*n-1);
    
    
    % z-update with relaxation (box constraint)
    zold = z;
    x_hat = alpha*x +(1-alpha)*zold;
    z = min(ub, max(lb, x_hat + u));

    % u-update
    u = u + (x_hat - z);        

  
    
     % diagnostics, reporting, termination checks
     X=vec2mat(x,n)';
     history.objval(k)  = objective(A,B, X);
% 
     history.r_norm(k)  = norm(x - z);
     history.s_norm(k)  = norm(-rho*(z - zold));
% 
     history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
     history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
% 
     if ~QUIET
         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\t%10.2f\n', k, ...
             history.r_norm(k), history.eps_pri(k), ...
             history.s_norm(k), history.eps_dual(k), history.objval(k),rho);
     end
% 
     if (history.r_norm(k) < history.eps_pri(k) && ...
        history.s_norm(k) < history.eps_dual(k))
          break;
     end
%     
%     % varying rho
%     if history.r_norm(k)>= 4*history.s_norm(k)
%         rho=2*rho;
%     end
%     if history.s_norm(k)>= 4*history.r_norm(k)
%             rho=rho/2;
%     end
        
                
    
end

if ~QUIET
    toc(t_start);
end
end


% define objective function 
function obj = objective(A,B,X)
    obj = 0.5*norm(A*X-X*B,'fro')^2;
end