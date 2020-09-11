%% graph matching ER case 

% extensive simulations
% Compare 3 methods: DP, QP, and SP


clear all;

%n_vec=[100,200,400,800,1600];
%n_vec=[500,1000,2000,4000,8000];
%n_vec=[30,60];
%n_vec=[30];
%n_vec=[10000];
%gamma_vec=[0.7,0.75,0.8,0.85,0.9,0.95];
%gamma_vec=[0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1];
%gamma_vec=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
gamma_vec=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
%p=0.1;
%gamma_vec=[0.9];    %sampling probability
%n_vec=[100,200,400,800,1000];
%n_vec=[400];
%gamma_vec=[0.8];
n_vec=[1000];
%n_vec=[100,200,400];
n_len=length(n_vec);
gamma_len=length(gamma_vec);


ind_run=10;
%ind_run=1;
k=1;
iter_max=100;


spec_corr=zeros(n_len,gamma_len,ind_run);
pp_corr=zeros(n_len,gamma_len,ind_run);
spec_corr_prob=zeros(n_len,gamma_len);
pp_corr_prob=zeros(n_len,gamma_len);

cvx_corr=spec_corr;
cvx_corr_prob=spec_corr_prob;
cvx_pp_corr=pp_corr;
cvx_pp_corr_prob=pp_corr_prob;


wass_corr=zeros(n_len,gamma_len,ind_run);
wass_corr_prob=zeros(n_len,gamma_len,ind_run);
wasss_pp_corr=wass_corr;
wass_pp_corr_prob=wass_corr_prob;

spec_run=zeros(n_len,gamma_len,ind_run);
cvx_run=spec_run;
wass_run=spec_run;




for i = 1:1:n_len
    n=n_vec(i);
  %  p=(log(n))^2/n;
  p=1/2;
    gamma_unorm=gamma_vec;
    %gamma_unorm=gamma_vec/log(n);
    for sim_ind=1:1:ind_run
    
        
        fprintf('The %i Independent run\n', sim_ind);
       
        %parent graph
        A=binornd(1,p,n,n);
        A=triu(A,1);
        A=A+A';
        
        for j = 1:1:gamma_len
            gamma=gamma_unorm(j);   
            s=1-gamma^2;      %gamma^2 = delta = 1-s
             %subsample graph
            Z1=binornd(1,s,n,n);
            Z1=triu(Z1,1);
            Z1=Z1+Z1';
            A1=A.*Z1; 
        
           perp_rnd=randperm(n);
        %perp_rnd=[1:1:n];
            S_rnd=zeros(n,n);
            S_rnd(1:1:n,perp_rnd)=eye(n);
            A_permuted=S_rnd*A*S_rnd';

            Z2=binornd(1,s,n,n);
            Z2=triu(Z2,1);
            Z2=Z2+Z2';
            A2=A_permuted.*Z2;
            
            W1=A1;
            W2=A2;
            
            % spectral method %%%%%%%%%%%%%%%%%%%%
            
            tic;
              [S0] = spectral_init(W1, W2);
              r0=full(sum(dot(S_rnd,S0))/n);
              spec_corr(i,j,sim_ind)= r0;
             spec_run (i, j, sim_ind)=toc;
             
             fprintf('Spectral Initial correct fraction is %4.2d\n', r0);
             
             
            % gradient descent
             r_old=r0;
             S=S0;    
             
            for iter_count=1:1:iter_max
                X=W2*S*W1;
               
                [S,] = greedy_match(X);              
                r=full(sum(dot(S_rnd,S))/n);
                if abs(r-r_old) <1e-4;  %convergence criterion
                    break;
                end
                r_old=r;
                fprintf('Correct fraction at iter %i is %4.2d\n', iter_count,r);
            end
            pp_corr(i,j,sim_ind)=r;
            
            
         
            
          % convex method %%%%%%%%%%
            tic;
            [z, history] = quadprog_admm(A1,A2, [], ones(2*n,1), zeros(n^2,1),ones(n^2,1), 4, 1.5,1e-5,1e-3);
            Xhat=vec2mat(z,n)';
            S_QP= greedy_match(Xhat');
            r0=full(sum(dot(S_rnd,S_QP))/n);
            cvx_corr(i,j,sim_ind)= r0;
              cvx_run(i,j,sim_ind)=toc;
             fprintf('Convex Initial correct fraction is %4.2d\n', r0);
       
             
            % gradient descent
             r_old=r0;
             S=S_QP;    
             
            for iter_count=1:1:iter_max
                X=W2*S*W1;
               
                [S,] = greedy_match(X);              
                r=full(sum(dot(S_rnd,S))/n);
                if abs(r-r_old) <1e-4;  %convergence criterion
                    break;
                end
                r_old=r;
                fprintf('Correct fraction at iter %i is %4.2d\n', iter_count,r);
            end
            cvx_pp_corr(i,j,sim_ind)=r;
    
            
            
                %Wasserstein method %%%%%%%%%%%%%%%
                tic;
                deg1=sum(W1);       % degree sequence
                deg2=sum(W2);
                [deg1_sort,ind1]=sort(deg1);
                [deg2_sort,ind2]=sort(deg2);

                 % degree profile (sorted)        
                 N_deg1=cell(n,1);
                 F_deg1=cell(n,1);
                N_deg2=N_deg1;
                F_deg2=F_deg1;
                for ind=1:1:n  
                    temp1=deg1_sort(logical(W1(ind,ind1)));
                    temp2=deg2_sort(logical(W2(ind,ind2))); 
                    [temp1_unique,temp1_a,temp1_c]= unique(temp1);
                    N_deg1{ind}=temp1_unique;
                    temp1_counts = accumarray(temp1_c,1);
                    F_deg1{ind}=temp1_counts/deg1(ind);
    
                    [temp2_unique,temp2_a,temp2_c]= unique(temp2);
                    N_deg2{ind}=temp2_unique;
                    temp2_counts = accumarray(temp2_c,1);
                    F_deg2{ind}=temp2_counts/deg2(ind); 
                end

                D=n*ones(n,n);
                for ind_i=1:1:n
                  if deg1(ind_i)>0
                    for ind_j=1:1:n 
                        if deg2(ind_j)>0    
                        D(ind_i,ind_j) =  dwass_discrete2(N_deg1{ind_i},N_deg2{ind_j},F_deg1{ind_i},F_deg2{ind_j});  
                            %Wasserstein distance
                        end
                    end
                   end
                end


            S_WASS= greedy_match((-1)*D');
              r0=full(sum(dot(S_rnd,S_WASS))/n);
              wass_corr(i,j,sim_ind)= r0;
              wass_run(i,j,sim_ind)=toc;
            fprintf('Wasserstein correct fraction is %4.2d\n', r0);
            
            % gradient descent
            
              r_old=r0;
              S=S_WASS;    
              
             for iter_count=1:1:iter_max
                 X=W2*S*W1;
                
                 [S,] = greedy_match(X);              
                 r=full(sum(dot(S_rnd,S))/n);
                 if abs(r-r_old) <1e-4;  %convergence criterion
                     break;
                 end
                 r_old=r;
                 fprintf('Correct fraction at iter %i is %4.2d\n', iter_count,r);
             end
             wass_pp_corr(i,j,sim_ind)=r;
             
             
            
        end
    end
       
end

for i = 1:1:n_len
     for  j = 1:1:gamma_len
         spec_corr_prob(i,j)=length(find(spec_corr(i,j,:)>0.95))/ind_run;
         pp_corr_prob(i,j)=length(find(pp_corr(i,j,:)>0.95))/ind_run;
         cvx_corr_prob(i,j)=length(find(cvx_corr(i,j,:)>0.95))/ind_run;
         cvx_pp_corr_prob(i,j)=length(find(cvx_pp_corr(i,j,:)>0.95))/ind_run;
           wass_corr_prob(i,j)=length(find(wass_corr(i,j,:)>0.95))/ind_run;
          wass_pp_corr_prob(i,j)=length(find(wass_pp_corr(i,j,:)>0.95))/ind_run;
     end
 end

 spec_corr_med=median(spec_corr,3);
 pp_corr_med=median(pp_corr,3);
 cvx_corr_med=median(cvx_corr,3);
 cvx_pp_corr_med=median(cvx_pp_corr,3);
wass_corr_med=median(wass_corr,3);
wass_pp_corr_med=median(wass_pp_corr,3);


spec_run_sum=sum(sum(sum(spec_run)));
cvx_run_sum=sum(sum(sum(cvx_run)));
wass_run_sum=sum(sum(sum(wass_run)));

save('ER_comparison_new.mat');



%% plot results
line_width=1.5;
Marker_size=6;
plot_spec={'k--*','k-*','r--d','r-d','b--o','b-o'};
leng_spec={'QP+','QP','DP+','DP','SP+','SP'};
i=1;
figure;
plot(gamma_vec, cvx_pp_corr_med(i,:), plot_spec{1},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, cvx_corr_med(i,:), plot_spec{2},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, wass_pp_corr_med(i,:), plot_spec{3},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, wass_corr_med(i,:), plot_spec{4},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, pp_corr_med(i,:), plot_spec{5},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, spec_corr_med(i,:), plot_spec{6},'LineWidth', line_width, 'MarkerSize', Marker_size );

legend(leng_spec,'location', 'best', 'FontSize', 20,'Interpreter','latex');
xlabel('$\sqrt{\delta}$','FontSize',20,'Interpreter','latex');
ylabel ('fraction of correctly matched pairs','FontSize',20,'Interpreter','latex');
xlim([0 0.8])
hFig1=gcf;


savefilename1='ER_comparison_new';
saveas(hFig1, ['./fig_files/',savefilename1], 'fig'); % save .fig files in a separate subfolder
addpath('./export_fig');
export_fig(savefilename1, '-pdf', '-transparent', hFig1);