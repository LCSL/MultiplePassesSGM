% Author: Raffaello Camoriano <raffaello.camoriano@iit.it>
% All rights reserved
% Project: "Generalization Properties and Implicit Regularization for Multiple Passes SGM", appearing in ICML 2016

setenv('LC_ALL','C');

enitorPath = '/home/kammo/Repos/Enitor/';   % Installation path of Enitor (https://github.com/Ommac/Enitor)
libsvmPath = '/home/kammo/Repos/libsvm/';   % Installation path of LIBSVM (https://github.com/cjlin1/libsvm)

% Import Enitor
addpath(genpath([enitorPath , 'algorithm']));
addpath(genpath([enitorPath , 'experiment']));
addpath(genpath([enitorPath , 'featureMap']));
addpath(genpath([enitorPath , 'filter']));
addpath(genpath([enitorPath , 'lossFunctions']));
addpath(genpath([enitorPath , 'stoppingRule']));
addpath(genpath([enitorPath , 'utils']));

% Import LIBSVM
addpath(genpath([libsvmPath , 'matlab']));

% Add example datasets
addpath(genpath('example_datasets'));

clearAllButBP;
close all;

%% Initialization

runLIBSVM = 1;

numRep = 10;	% Number of repetitions

% Store preformance and predictions
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 1;
storeFullTrainPred = 0;
storeFullValPred = 0;
storeFullTestPred = 0;

verbose = 0;
saveResult = 0;

% Set experimental results relative directory name
resdir = 'results/';
mkdir(resdir);

epochs = 200;
validationPart = 0.2;
errorStorageMode = 'epochs';

% Use hinge loss to reproduce the plots in the paper
% lossFunction = @hingeLoss;    

% Use hinge loss to compare with LIBSVM in terms of accuracy
lossFunction = @classificationError;

%%%%%%%%%%%%%%%%%%%%%%%%
% SGD eta-theta ranges %
%%%%%%%%%%%%%%%%%%%%%%%%

%%% ETA

etaGuesses = logspace(-2,0,20);

%%% THETA

% Range beta \in [0,1]
% First half: beta/(beta + 1); second half: 1/(beta + 1)
betaGuesses = linspace(0,1,11);
thetaGuesses = betaGuesses ./ (betaGuesses + 1);
thetaGuesses = [thetaGuesses , flip(1 ./ (betaGuesses(1:end-1) + 1))];

%%%%%%%%%%%%%%%%%%%%%%%%

% Map
precomputeKernel = 1;
map = @gaussianKernel;

% Dataset
datasetRef = @Adult;
coding = 'plusMinusOne';
ntrds = 1000;
nteds = 16282;
mapParGuesses = 4;

ntr = floor(ntrds * (1-validationPart));

shuffleTraining = 1;
shuffleTest = 1;
shuffleAll = 1;

%% Storage vars init

results = struct();

SGD_CV_1_results = struct();
SGD_CV_1_results.trainTime = zeros(numRep,1);
SGD_CV_1_results.testTime = zeros(numRep,1);
SGD_CV_1_results.perf = zeros(numRep,1);
SGD_CV_1_results.trainErr = zeros(numRep,numel(etaGuesses));
SGD_CV_1_results.valErr = zeros(numRep,numel(etaGuesses));
SGD_CV_1_results.testErr = zeros(numRep,numel(etaGuesses));

SGD_CV_2_results = struct();
SGD_CV_2_results.trainTime = zeros(numRep,1);
SGD_CV_2_results.testTime = zeros(numRep,1);
SGD_CV_2_results.perf = zeros(numRep,1);
SGD_CV_2_results.trainErr = zeros(numRep,numel(thetaGuesses));
SGD_CV_2_results.valErr = zeros(numRep,numel(thetaGuesses));
SGD_CV_2_results.testErr = zeros(numRep,numel(thetaGuesses));

SIGD_RR_norep_1_results = struct();
SIGD_RR_norep_1_results.trainTime = zeros(numRep,1);
SIGD_RR_norep_1_results.testTime = zeros(numRep,1);
SIGD_RR_norep_1_results.perf = zeros(numRep,1);
SIGD_RR_norep_1_results.trainErr = [];
SIGD_RR_norep_1_results.valErr = [];
SIGD_RR_norep_1_results.testErr = [];

SIGD_RR_norep_2_results = struct();
SIGD_RR_norep_2_results.trainTime = zeros(numRep,1);
SIGD_RR_norep_2_results.testTime = zeros(numRep,1);
SIGD_RR_norep_2_results.perf = zeros(numRep,1);
SIGD_RR_norep_2_results.trainErr = [];
SIGD_RR_norep_2_results.valErr = [];
SIGD_RR_norep_2_results.testErr = [];

%% Experiment loop

for k = 1:numRep

    % Load dataset
    ds = datasetRef(ntrds, nteds, coding, shuffleTraining, shuffleTest, shuffleAll);
    ds.lossFunction = lossFunction;
    
    %% Run LibSVM with C & gamma cross-validation
    
    if runLIBSVM == 1

        % Training
        tic
        bestcv = 0;
        for log2c = -1:3,
            for log2g = -4:1,
                cmd = ['-m 2000 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
                model = svmtrain( ds.Y(ds.trainIdx(1:ntr)) , ds.X(ds.trainIdx(1:ntr),:),  cmd);

                [predict_label, accuracy, dec_values] = ...
                    svmpredict(ds.Y(ds.trainIdx((ntr+1):ntrds)), ds.X(ds.trainIdx((ntr+1):ntrds),:), model);            

                if (accuracy(1) >= bestcv),
                    bestcv = accuracy(1); bestc = 2^log2c; bestg = 2^log2g;
                    bestModel = model;
                end
            end
        end
        LIBSVM_results.trainTime(k) = toc;

        % Testing
        tic
        [predicted_labels, accuracy, ~] = ...
            svmpredict(ds.Y(ds.testIdx,:), ds.X(ds.testIdx,:), bestModel);    

        LIBSVM_results.testTime(k) = toc;

        LIBSVM_results.perf(k) = 1 - accuracy(1) /100;
    end
    
    %% Run SGD - fixed theta = 0
    
    fil = @SsubGD_dual_hinge_loss;
    alg = ksgdesc( map , fil , ...
        'mapParGuesses' , mapParGuesses , ...
        'verbose' , 0 , ...
        'etaGuesses' , etaGuesses , ...
        'thetaGuesses' , 0 , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf , ...
        'storeFullTrainPred' , storeFullTrainPred, ...
        'storeFullValPred' , storeFullValPred , ...
        'storeFullTestPred' , storeFullTestPred);

    expSGD_CV_1_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);

    expSGD_CV_1_kernel_hinge_loss.run();
    expSGD_CV_1_kernel_hinge_loss.result

    SGD_CV_1_results.trainTime(k) = expSGD_CV_1_kernel_hinge_loss.time.train;
    SGD_CV_1_results.testTime(k) = expSGD_CV_1_kernel_hinge_loss.time.test;
    SGD_CV_1_results.perf(k) = expSGD_CV_1_kernel_hinge_loss.result.perf;
    SGD_CV_1_results.trainErr(k,:) = expSGD_CV_1_kernel_hinge_loss.algo.trainPerformance;
    SGD_CV_1_results.valErr(k,:) = expSGD_CV_1_kernel_hinge_loss.algo.valPerformance;
    SGD_CV_1_results.testErr(k,:) = expSGD_CV_1_kernel_hinge_loss.algo.testPerformance;

    results(k).SGD_CV_1 = expSGD_CV_1_kernel_hinge_loss.result;

    %% Run SGD - fixed eta = 1/4
        
    fil = @SsubGD_dual_hinge_loss;
    alg = ksgdesc( map , fil , ...
        'mapParGuesses' , mapParGuesses , ...
        'verbose' , 0 , ...
        'etaGuesses' , 1/4 , ...
        'thetaGuesses' , thetaGuesses , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf , ...
        'storeFullTrainPred' , storeFullTrainPred, ...
        'storeFullValPred' , storeFullValPred , ...
        'storeFullTestPred' , storeFullTestPred);

    expSGD_CV_2_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);

    expSGD_CV_2_kernel_hinge_loss.run();
    expSGD_CV_2_kernel_hinge_loss.result

    SGD_CV_2_results.trainTime(k) = expSGD_CV_2_kernel_hinge_loss.time.train;
    SGD_CV_2_results.testTime(k) = expSGD_CV_2_kernel_hinge_loss.time.test;
    SGD_CV_2_results.perf(k) = expSGD_CV_2_kernel_hinge_loss.result.perf;
    SGD_CV_2_results.trainErr(k,:) = expSGD_CV_2_kernel_hinge_loss.algo.trainPerformance;
    SGD_CV_2_results.valErr(k,:) = expSGD_CV_2_kernel_hinge_loss.algo.valPerformance;
    SGD_CV_2_results.testErr(k,:) = expSGD_CV_2_kernel_hinge_loss.algo.testPerformance;

    results(k).SGD_CV_2 = expSGD_CV_2_kernel_hinge_loss.result;
    
    

    %% Run SIGD_RR_norep , fixed eta = 1/sqrt(m) & fixed theta = 0

    fil = @SIsubGD_dual_hinge_loss;
    maxiter2 = ntr * epochs;
    alg = kigdesc( map , fil , ...
        'mapParGuesses' , mapParGuesses , ...
        'numFilterParGuesses' , maxiter2   , ...
        'precomputeKernel' , precomputeKernel , ...
        'ordering' , 'reshuffle_norep' , ...
        'eta' , 1/sqrt(ntr) , ...
        'theta' , 0, ...
        'verbose' , 0 , ...
        'errorStorageMode' , errorStorageMode , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf , ...
        'storeFullTrainPred' , storeFullTrainPred, ...
        'storeFullValPred' , storeFullValPred , ...
        'storeFullTestPred' , storeFullTestPred);

    expSIGD_RR_norep_1_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expSIGD_RR_norep_1_kernel_hinge_loss.run();
    expSIGD_RR_norep_1_kernel_hinge_loss.result
    
    SIGD_RR_norep_1_results.trainTime(k) = expSIGD_RR_norep_1_kernel_hinge_loss.time.train;
    SIGD_RR_norep_1_results.testTime(k) = expSIGD_RR_norep_1_kernel_hinge_loss.time.test;
    SIGD_RR_norep_1_results.perf(k) = expSIGD_RR_norep_1_kernel_hinge_loss.result.perf;
    SIGD_RR_norep_1_results.trainErr = [SIGD_RR_norep_1_results.trainErr , expSIGD_RR_norep_1_kernel_hinge_loss.algo.trainPerformance'];
    SIGD_RR_norep_1_results.valErr = [SIGD_RR_norep_1_results.valErr , expSIGD_RR_norep_1_kernel_hinge_loss.algo.valPerformance'];
    SIGD_RR_norep_1_results.testErr = [SIGD_RR_norep_1_results.testErr , expSIGD_RR_norep_1_kernel_hinge_loss.algo.testPerformance'];

    results(k).SIGD_RR_norep_1 = expSIGD_RR_norep_1_kernel_hinge_loss.result;

    %% Run SIGD_RR_norep, fixed eta = 1/4, fixed theta = 1/2

    fil = @SIsubGD_dual_hinge_loss;
    maxiter2 = ntr * epochs;
    alg = kigdesc( map , fil , ...
        'mapParGuesses' , mapParGuesses , ...
        'numFilterParGuesses' , maxiter2   , ...
        'precomputeKernel' , precomputeKernel , ...
        'ordering' , 'reshuffle_norep' , ...
        'eta' , 1/4 , ...
        'theta' , 1/2, ...
        'verbose' , 0 , ...
        'errorStorageMode' , errorStorageMode , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf , ...
        'storeFullTrainPred' , storeFullTrainPred, ...
        'storeFullValPred' , storeFullValPred , ...
        'storeFullTestPred' , storeFullTestPred);

    expSIGD_RR_norep_2_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expSIGD_RR_norep_2_kernel_hinge_loss.run();
    expSIGD_RR_norep_2_kernel_hinge_loss.result
    
    SIGD_RR_norep_2_results.trainTime(k) = expSIGD_RR_norep_2_kernel_hinge_loss.time.train;
    SIGD_RR_norep_2_results.testTime(k) = expSIGD_RR_norep_2_kernel_hinge_loss.time.test;
    SIGD_RR_norep_2_results.perf(k) = expSIGD_RR_norep_2_kernel_hinge_loss.result.perf;
    SIGD_RR_norep_2_results.trainErr = [SIGD_RR_norep_2_results.trainErr , expSIGD_RR_norep_2_kernel_hinge_loss.algo.trainPerformance'];
    SIGD_RR_norep_2_results.valErr = [SIGD_RR_norep_2_results.valErr , expSIGD_RR_norep_2_kernel_hinge_loss.algo.valPerformance'];
    SIGD_RR_norep_2_results.testErr = [SIGD_RR_norep_2_results.testErr , expSIGD_RR_norep_2_kernel_hinge_loss.algo.testPerformance'];

    results(k).SIGD_RR_norep_2 = expSIGD_RR_norep_2_kernel_hinge_loss.result;
end

%% Save workspace

save([resdir , 'workspace_LCSL_01.mat']);


%% Plots

% Best test accuracy

if strcmp(func2str(lossFunction), 'classificationError') && ...
        runLIBSVM == 1
    
    figure
    hold on
    title({'Test Accuracy Comparison';'SGD vs SIGD vs LIBSVM'})
    if numRep == 1
        bar([SGD_CV_1_results.perf , ...
            SGD_CV_2_results.perf , ...
            SIGD_RR_norep_1_results.perf , ...
            SIGD_RR_norep_2_results.perf , ...
            LIBSVM_results.perf']);
    else
        boxplot([SGD_CV_1_results.perf , ...
            SGD_CV_2_results.perf , ...
            SIGD_RR_norep_1_results.perf , ...
            SIGD_RR_norep_2_results.perf , ...
            LIBSVM_results.perf'] , 'labels' , {'SGM Const', 'SGM Decay', 'SIGD Const', 'SIGD Decay', 'LIBSVM'});
    end
    xlabel('Algorithm')
    ylabel('Test Error')
    hold off    
end




% SGD CV - fixed theta = 0

figure
hold on
title({'SGM CV - Fixed \theta = 0, CV eta \in (0,1]';'Training vs. Test Error'})
if numRep == 1
    h1 = plot(1./etaGuesses,SGD_CV_1_results.trainErr);
    h2 = plot(1./etaGuesses,SGD_CV_1_results.testErr);
    set(gca, 'xscale', 'log')
else
    h1 = bandplot( 1./etaGuesses , SGD_CV_1_results.trainErr , 'red' , 0.1 , 1 , 1, '-');
    h2 = bandplot( 1./etaGuesses , SGD_CV_1_results.testErr , 'blue' , 0.1 , 1 , 1, '-');
end
xlabel('1/\eta_1')
ylabel('Error')
legend([h1,h2],'Training','Test')
hold off


% SGD CV - fixed eta = 1/4

figure
hold on
title({'SGM CV - Fixed \eta = 1/4, CV theta \in [0,1]';'Training vs. Test Error'})
if numRep == 1
    h1 = plot(thetaGuesses,SGD_CV_2_results.trainErr);
    h2 = plot(thetaGuesses,SGD_CV_2_results.testErr);
else
    h1 = bandplot( thetaGuesses , SGD_CV_2_results.trainErr , 'red' , 0.1 , 0 , 1, '-');
    h2 = bandplot( thetaGuesses , SGD_CV_2_results.testErr , 'blue' , 0.1 , 0 , 1, '-');
end
xlabel('\theta')
ylabel('Error')
legend([h1,h2],'Training','Test')
hold off


% SIGD_RR_norep , fixed eta = 1/sqrt(m) & fixed theta = 0

axes5 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
box(axes5,'on');
hold on;
if numRep == 1
    h1 = plot(SIGD_RR_norep_1_results.trainErr);
    h3 = plot(SIGD_RR_norep_1_results.testErr);
else
    h1 = bandplot( [] , SIGD_RR_norep_1_results.trainErr' , 'red' , 0.1 , 0 , 1, '-');
    h3 = bandplot( [] , SIGD_RR_norep_1_results.testErr' , 'blue' , 0.1 , 0 , 1, '-');
end
title({'Stochastic IGM - Random reshuffling with no repetitions';'fixed \eta = 1/sqrt(m) & fixed \theta = 0';'Empirical vs. Generalization Error'})
legend([h1,h3] , 'Training','Test')
xlabel('Passes')
ylabel('Error')
hold off


% SIGD_RR_norep, fixed eta = 1/4, fixed theta = 1/2

axes6 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
box(axes6,'on');
hold on;
if numRep == 1
    h1 = plot(SIGD_RR_norep_2_results.trainErr);
    h3 = plot(SIGD_RR_norep_2_results.testErr);
else
    h1 = bandplot( [] , SIGD_RR_norep_2_results.trainErr' , 'red' , 0.1 , 0 , 1, '-');
    h3 = bandplot( [] , SIGD_RR_norep_2_results.testErr' , 'blue' , 0.1 , 0 , 1, '-');
end
title({'Stochastic IGM - Random reshuffling with no repetitions';'fixed \eta = 1/4, fixed \theta = 1/2';'Empirical vs. Generalization Error'})
legend([h1,h3] , 'Training','Test')
xlabel('Passes')
ylabel('Error')
hold off
