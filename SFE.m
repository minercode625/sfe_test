function stats = SFE( IDATA, IANSWER,IFFCS, ILCONS, IPERF)

global data
global answer
global col
global lcol
global row
global lcons
global acalls
global perf
global lacc_mat
data = IDATA;
answer = IANSWER;
ffcs = IFFCS;
acalls = 0;
lcons = ILCONS;
perf = IPERF;
lcol = size(answer,2);
[row,col] = size( data );

rng(0)
UR=0.3;
UR_Max=0.3;
UR_Min=0.001;
lacc_mat = [];

while acalls < ffcs

    EFs=0;
    X=randi([0,1],1,col);      % Initialize an Individual X
    s_size = sum(X);
    if s_size > lcons
        samp_size = s_size - lcons;
        idx1 = find(X==1);
        t_idx = randsample(idx1,samp_size);
        X(1,t_idx) = 0;
    end
    Fit_X=evaluate(X);           % Calculate the Fitness of X

    while EFs < 10
        EFs=EFs+1;
        X_New=X;

        %% Non-selection operation
        [~,U_Index]=find(X==1);          % Find Selected Features in X
        NUSF_X=size(U_Index,2);          % Number of Selected Features in X
        UN=ceil(UR*col);                % The Number of Features to Unselect: Eq(2)
        %SF=randperm(20,1);              % The Number of Features to Unselect: Eq(6)
        K1=randi(NUSF_X,UN,1);           % Generate UN random number between 1 to the number of slected features in X
        K=U_Index(K1)';                  % K=index(U)
        X_New(K)=0;                      % Set X_New (K)=0



     %% Selection operation
        if sum(X_New)==0
            [~,S_Index]=find(X==0);     % Find non-selected Features in X
            NSF_X=size(S_Index,2);      % Number of non-selected Features in X
            SN=1;                       % The Number of Features to Select
            K1=randi(NSF_X,SN,1);       % Generate SN random number between 1 to the number of non-selected features in X
            X_New=X;
            K=S_Index(K1)';
            X_New(K)=1;                 % Set X_New (K)=1
        end
        s_size = sum(X_New);
        if s_size > lcons
            samp_size = s_size - lcons;
            idx1 = find(X_New==1);
            t_idx = randsample(idx1,samp_size);
            X_New(1,t_idx) = 0;
        end

        Fit_X_New=evaluate(X_New); %Calculate the Fitness of X_New

        if Fit_X_New<Fit_X
            X=X_New;
            Fit_X=Fit_X_New;

        end

        UR=(UR_Max-UR_Min)*((ffcs-EFs)/ffcs)+UR_Min;  % Eq(3)
        EFs = EFs+1;
        if acalls >= ffcs
            break;
        end
    end

end

opt_vec = X;
stats = cell(1,2);
stats{1,1} = opt_vec;
stats{1,2} = lacc_mat;
end

function [val, lacc] = evaluate( chr )
% Increase the number of actual fitness function calls

global acalls
acalls = acalls + 1;
if all(chr==0)
    val = inf;
    return;
end

global data
global row
global answer
global perf
global lcol
global lacc_mat

val = 0;

[train,test] = crossvalind( 'holdout', ones(row,1), 0.2 );
[pre,post] = pmlbayes_matlab( data(train,chr==1), answer(train,:), data(test,chr==1));

inter = answer(test,:) == pre;
lacc = sum(inter,1) / size(pre,1);
lacc= sort(lacc);
target_ffc = [50	61	72	83	94	105	116	127	138	149	160	171	182	193	204	215	226	237	248	259	270	281	292	300];
l_50 = ceil(lcol/2);
l_25 = ceil(lcol/4);

acc_50 = mean(lacc(1:l_50));
acc_25 = mean(lacc(1:l_25));

if acalls == 1
    a = 1;
end
%check if the current ffc is in the target ffc list
if ismember(acalls, target_ffc)
    %if yes, then save the current lacc
    lacc_mat(end+1, :) = [acc_50, acc_25];
end



if strcmp( perf, 'hloss' )
    val = hloss( answer(test,:), pre );
elseif strcmp( perf, 'rloss' )
    val = rloss( answer(test,:), post );
elseif strcmp( perf, 'mlacc' )
    [ta,~,~,~] = mlacc( answer(test,:), pre );
    val = val - ta;
elseif strcmp( perf, 'setacc' )
    val = -setacc( answer(test,:), pre );
elseif strcmp( perf, 'onerr' )
    val = onerr( answer(test,:), post );
elseif strcmp( perf, 'mlcov' )
    val = mlcov( answer(test,:), post );
end
end

%% MLNB
function [pre,post] = pmlbayes_matlab( train, answer, test )
% Multi Label Naive Bayes
lcol = size( answer, 2 );
pre = zeros( size(test,1), lcol );
post = zeros( size(test,1), lcol );

for k=1:lcol
    model = fitcnb( train, answer(:,k),  'DistributionNames', 'mvmn' );

    [pre(:,k),t] = predict( model, test  );
    t(isnan(t(:,end)),end) = 0;
    pre(isnan(pre(:,k)),k) = 0;

    post(:,k) = t(:,end);
end
end
