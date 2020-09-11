%% graph matching Wigner case 

% extensive simulations
% Compare 3 methods: DP, QP, and SP

% W2 = sqrt{ 1- sigma^2} W1 + sigma*Z

clear all;

%n_vec=[100,200,400,800,1600];
%n_vec=[500,1000,2000,4000,8000];
%n_vec=[30,60];
%n_vec=[30];
%n_vec=[10000];
gamma_vec=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
%gamma_vec=[0.5,0.6,0.7,0.8,0.9,1.0];
%gamma_vec=[0.72,0.74,0.76,0.78];
%n_vec=[100,200,400,800];
%n_vec=[1000];
n_vec=[1000];
%gamma_vec=[0.05];
n_len=length(n_vec);
gamma_len=length(gamma_vec);


ind_run=10;
%ind_run=1;
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
    gamma_unorm=gamma_vec;
  
    for sim_ind=1:1:ind_run
    
        
        fprintf('The %i Independent run\n', sim_ind);
       
        W1=normrnd(0,1,n,n);
        W1=(W1+W1')/sqrt(2);
        perp_rnd=randperm(n);
        S_rnd=zeros(n,n);
        S_rnd(1:1:n,perp_rnd)=eye(n);
        Z=normrnd(0,1,n,n);
        Z=(Z+Z')/sqrt(2);        
       
        
        for j = 1:1:gamma_len
            gamma=gamma_unorm(j);       
            W2=sqrt(1-gamma^2)*S_rnd*W1*S_rnd'+gamma*Z;
            
            
            % spectral method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            
            
        
            % convex method %%%%%%%%%%%%%%%%%%%%%%%
            tic;
            [z, history] = quadprog_admm(W1,W2, [], ones(2*n,1), zeros(n^2,1),ones(n^2,1), 40, 1.5,1e-5,1e-3);

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
            
            
            
             % Wass method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                tic;
                D=zeros(n,n);
                W1_sort=sort(W1); %sort columns
                W2_sort=sort(W2);

                for ind_i=1:1:n
                    for ind_j=1:1:n
                        D(ind_i,ind_j) = sum(abs(W1_sort(:,ind_i)-W2_sort(:,ind_j)))/n;    
                        %Wasserstein distance
                    end
                end

                S_WASS= greedy_match((ones(n,n)-D)');
                r0=full(sum(dot(S_rnd,S_WASS))/n);
                wass_corr(i,j,sim_ind)=r0;
                wass_run(i,j,sim_ind)=toc;
               fprintf('Wasserstein correct fraction is %4.2d\n',r0);

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

save('Wigner_comparison_new.mat');

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
xlabel('$\sigma$','FontSize',20,'Interpreter','latex');
ylabel ('fraction of correctly matched pairs','FontSize',20,'Interpreter','latex');
hFig1=gcf;



savefilename1='Wigner_comparison_new';
saveas(hFig1, ['./fig_files/',savefilename1], 'fig'); % save .fig files in a separate subfolder
addpath('./export_fig');
export_fig(savefilename1, '-pdf', '-transparent', hFig1);