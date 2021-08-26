%% Tchagang Alain B.
%
clear
%% Reading Preprocessed QM7 data from files
[data,text] = xlsread('QM7_Molecules_Atoms_1Property_AE.xlsx'); % load atomic composition (AC) and Atomization Energy
U = data(:,1:5);
AE = data(:,6); % Atomization Energy Property
Molecules = text(2:end,1); % QM7 molecule names
[x,id] = xlsread('Cmat_Results.xlsx'); % Load sorted Coulomb matrix (SCM)
[z, idEig] = xlsread('qm7Molecule_Eigen-Spectrum_AE.xlsx'); % Load Coulomb eigen spectrum (CES)

%% Define input and Output

% Chose input
% X = z; % Coulomb eigen spectrum (CES)
% X = x; % Sorted Coulomb matrix (SCM)
% X = [U z]; % Atomic Composition (AC) and Coulomb eigen spectrum (CES)
 X = [U x]; % Atomic Composition (AC) and Sorted Coulomb matrix (SCM)

% initialize the output and take the zscore to normalize the data
[yo, aemean, aestd] = zscore(AE);


%% Neural Network

x = X'; %   input data.
t = yo'; %   target data

trainFcn = 'trainbr';  % Bayesian Regularization.
%trainFcn = 'trainscg';  % .
%trainFcn = 'trainlm';

% Create a Fitting Network
%hiddenLayerSize = [17]
%hiddenLayerSize = [16 8 4]
hiddenLayerSize = [18 9 3]; 

net = fitnet(hiddenLayerSize,trainFcn);
net.trainParam.epochs = 100; % can increase but time consuming

% Setup Division of Data for Training, Validation, Testing
% Selection is random results may slightly change
net.divideParam.trainRatio = 95/100; 
net.divideParam.valRatio = 3/100;
net.divideParam.testRatio = 2/100;

% Train the Network
[net,tr] = train(net,x,t);

%% Test the Network
yNN1 = net(x);
yNN = yNN1*aestd+aemean; % Unnormalized

%% Statistics
errorNN = gsubtract(t*aestd+aemean,yNN);
performance = perform(net,t*aestd+aemean,yNN);
rmseNN = sqrt(mean(errorNN.^2));
maeNN = mean(abs(errorNN));
R2NN = corr(yo,yNN');

%% View the Network
view(net)

%% Plots
figure, scatter(yo,yNN','x', 'LineWidth',2); grid; xlabel('Predictions'); ylabel('True AE'); title('Neural Network');
