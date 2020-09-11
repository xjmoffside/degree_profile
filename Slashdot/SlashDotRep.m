%% some global parameters
clear all;
seed = 13;
rng(seed);
% number of nodes
n = 750; 
% subsampling rate
%ss = [0.975, 0.95, 0.925, 0.9, 0.875, 0.85,0.825,0.8];
%ss=[0.85];
%ss=[0.825,0.8,0.775,0.75,0.725,0.7];
%ss=[0.825,0.8,0.775,0.75];
%ss=[0.625];
ss=[0.975:-0.025:0.6];
nss = numel(ss);
% # repetition at each rate
nrep = 10;
%nrep=1;
% one of the two slashdot datasets
filename = 'SD0811.txt';
iter_max=100;


%% read in a subset of SlashDot data

A = zeros(n,n);

g = fopen(filename);
cont = 1;
while cont
    txt = fgetl(g);
    txtnum = str2num(txt) + 1;
    if (max(txtnum) <= n && txtnum(1) ~= txtnum(2))
        A(txtnum(1), txtnum(2)) = 1;
        A(txtnum(2), txtnum(1)) = 1;
    end
    if (min(txtnum) > n)
        cont = 0;
    end
end
fclose(g);

%% define error matrices

corrDeg = zeros(nrep, nss);
corrQP = corrDeg;
corrSp = corrDeg;
corrDeg_pp=corrDeg;
corrQP_pp=corrDeg;
corrSP_pp=corrDeg;
DP_run=zeros(nrep,nss);
QP_run=DP_run;
SP_run=DP_run;
%% subsample edges to create two correlated binary matrices

for jj = 1:nss
    s = ss(jj);
    
    for ii = 1:nrep
        
        % sample two graphs to be matched
        mask1 = binornd(1, s, n, n);
        mask1 = triu(mask1, 1);
        mask1 = mask1 + mask1';
        
        mask2 = binornd(1, s, n, n);
        mask2 = triu(mask2, 1);
        mask2 = mask2 + mask2';
        
        W1 = A .* mask1;
        perp_rnd=randperm(n);
        S_rnd=zeros(n,n);
        S_rnd(1:1:n,perp_rnd)=eye(n);      
        W2 = S_rnd*A*S_rnd'.* mask2;
        
        %% degree profile matching
        
        % compute Wasserstein distance
        
        tic;
        deg1=sum(W1);       % degree sequence
        deg2=sum(W2);
        
        %[deg1_sort,ind1]=sort(deg1,'descend');
        %[deg2_sort,ind2]=sort(deg2,'descend');
        
        [deg1_sort,ind1]=sort(deg1);
        [deg2_sort,ind2]=sort(deg2);
        
        
        DP1=cell(n,1);      % degree profile (sorted)
        DP2=DP1;
        
        N_deg1=cell(n,1);
        F_deg1=cell(n,1);
        N_deg2=N_deg1;
        F_deg2=F_deg1;
        
        
        for i=1:1:n
            temp1=deg1_sort(logical(W1(i,ind1)));
            temp2=deg2_sort(logical(W2(i,ind2)));
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
        
        
        S_WASS= greedy_match((-1)*D');
        WASS_corr=full(sum(dot(S_rnd,S_WASS))/n);
        fprintf('Wasserstein correct fraction is %4.2d\n', WASS_corr);
        %toc;
        DP_run(ii,jj)=toc;
          %spy(S_WASS);
        corrDeg(ii,jj) = WASS_corr;
        
        % gradient descent
            r0=WASS_corr;
            S0=S_WASS;
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
            corrDeg_pp(ii,jj)=r;
            
            
      
        
        %% comparison with QP
        tic;
        [z, history] = quadprog_admm(W1,W2, [], ones(2*n,1), zeros(n^2,1),...
            ones(n^2,1), 40, 1.5,1e-5,1e-3);
        Xhat=vec2mat(z,n)';
        
        S_QP= greedy_match(Xhat');
        cvx_corr=full(sum(dot(S_rnd,S_QP))/n);
        fprintf('QP correct fraction is %4.2d\n', cvx_corr);
        corrQP(ii,jj) = cvx_corr;
        QP_run(ii,jj)=toc;
        
        % gradient descent
              r0=cvx_corr;
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
            corrQP_pp(ii,jj)=r;
        
        %% comparision with Spectral
        % W1 = A1;
        % W2 = A2;
        tic;
        S_spectral = spectral_init(W1, W2);
        spec_corr=full(sum(dot(S_rnd,S_spectral))/n);
        fprintf('Spectral correct fraction is %4.2d\n', spec_corr);
        corrSp(ii,jj) = spec_corr;
        SP_run(ii,jj)=toc;
        
         % gradient descent
             r_old=spec_corr;
             S=S_spectral;    
             
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
            corrSP_pp(ii,jj)=r;
        
    end
end

beep
pause(2)
beep
pause(2)
beep

%% make plots

mcorrDeg = median(corrDeg, 1);
mcorrQP = median(corrQP, 1);
mcorrSp = median(corrSp, 1);

DP_time=sum(sum(DP_run));
QP_time=sum(sum(QP_run));
SP_time=sum(sum(SP_run));

save('SlashDot.mat');

% %% plotting the mismatch patterns
% 
% fig = figure;
% subplot(2,3,1);
% spy(S_WASS);
% subplot(2,3,2);
% spy(S_QP);
% subplot(2,3,3);
% spy(S_spectral);
% subplot(2,3,4);
% spy(W1);
% subplot(2,3,5);
% spy(W2);
% subplot(2,3,6);
% spy(A);
% print(fig,strcat('MismatchSD','_n',num2str(n),'_rate',num2str(100*s),...
%     '_seed',num2str(seed)),'-dpdf');



%% plot results
line_width=1.5;
Marker_size=6;
plot_spec={'k--*','k-*','r--d','r-d','b--o','b-o'};
leng_spec={'QP+','QP','DP+','DP','SP+','SP'};
i=1;
gamma_vec=ss;
figure;
plot(gamma_vec, corrQP_pp(i,:), plot_spec{1},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, corrQP(i,:), plot_spec{2},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, corrDeg_pp(i,:), plot_spec{3},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, corrDeg(i,:), plot_spec{4},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, corrSP_pp(i,:), plot_spec{5},'LineWidth', line_width, 'MarkerSize', Marker_size );
hold on;
plot(gamma_vec, corrSp(i,:), plot_spec{6},'LineWidth', line_width, 'MarkerSize', Marker_size );

legend(leng_spec,'location', 'best', 'FontSize', 20,'Interpreter','latex');
xlabel('$s$','FontSize',20,'Interpreter','latex');
ylabel ('fraction of correctly matched pairs','FontSize',20,'Interpreter','latex');
xlim([0.8,0.975]);
xticks(gamma_vec);
hFig1=gcf;


savefilename1='SlashDot';
saveas(hFig1, ['./fig_files/',savefilename1], 'fig'); % save .fig files in a separate subfolder
addpath('./export_fig');
export_fig(savefilename1, '-pdf', '-transparent', hFig1);
exit;
