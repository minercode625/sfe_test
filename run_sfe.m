clear;
data_path = './Dataset/';
list = dir(data_path);
list = list(3:end);

data_path = './Dataset/';
dataName = 'arts.mat';
dataName
load(sprintf('%s%s',data_path,dataName));

if exist('data_dis', 'var') == 1
    data = data_dis;
end

exp_iter = 10;

parfor k=1:exp_iter
    train_data = data( sim_seq(:,k), : );
    train_answer = answer( sim_seq(:,k), : );
    
    fprintf('onerr\n');
    
    stats_SFE_onerr{k,1} = SFE( train_data, train_answer, 300, 50, 'onerr');
    [~,post] = pmlbayes_matlab( train_data(:,stats_SFE_onerr{k,1}{1,1}==1), ...
        train_answer, data( ~sim_seq(:,k),stats_SFE_onerr{k,1}{1,1}==1 ) );
    perf_SFE(k).onerr(1,1) = onerr( answer(~sim_seq(:,k), :), post );
    
    fprintf('mlacc\n');
    
    stats_SFE_mlacc{k,1} = SFE( train_data, train_answer, 300, 50, 'mlacc');
    [pre,~] = pmlbayes_matlab( train_data(:,stats_SFE_mlacc{k,1}{1,1}==1), ...
        train_answer, data( ~sim_seq(:,k),stats_SFE_mlacc{k,1}{1,1}==1 ) );
    perf_SFE(k).mlacc(1,1) = mlacc( answer(~sim_seq(:,k), :), pre );
    
    fprintf('rloss\n');
    
    stats_SFE_rloss{k,1} = SFE( train_data, train_answer, 300, 50, 'rloss');
    [~,post] = pmlbayes_matlab( train_data(:,stats_SFE_rloss{k,1}{1,1}==1), ...
        train_answer, data( ~sim_seq(:,k),stats_SFE_rloss{k,1}{1,1}==1 ) );
    perf_SFE(k).rloss(1,1) = rloss( answer(~sim_seq(:,k), :), post );
    
    fprintf('mlcov\n');
    
    stats_SFE_mlcov{k,1} = SFE( train_data, train_answer, 300, 50, 'mlcov');
    [~,post] = pmlbayes_matlab( train_data(:,stats_SFE_mlcov{k,1}{1,1}==1), ...
        train_answer, data( ~sim_seq(:,k),stats_SFE_mlcov{k,1}{1,1}==1 ) );
    perf_SFE(k).mlcov(1,1) = mlcov( answer(~sim_seq(:,k), :), post );
    
end

out_path = './';
save(sprintf('%s%s',out_path,dataName))

