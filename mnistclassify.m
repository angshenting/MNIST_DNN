% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

addpath('./data')
addpath('./exec')

clear all
close all

% Log name
logname = 'classify1';

% Logfile
logfile = [logname '.log'];

% Savefiles
weights_file = [logname '_weights'];
error_file = [logname '_errors'];
times_file = [logname '_times'];

% Batching Parameters
n_classes = 10;
batchsize = 100;


% RBM pretraining parameters
rbm_epoch = [100 100]; % No of epochs
layer_nodes = [512 1024]; % Configuration of layers

epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;
momentumchangeat = 6;

% Backpropagation parameters
backprop_epoch = 200;
SIG = 0.1; % Learning rate

%%%% LOG FILE %%%%

log_id = fopen(logfile,'w');

% Print variables to logfile
fprintf(log_id,'Layers + nodes: %d\n',layer_nodes);
fprintf(log_id,'Batch size: %d\n',batchsize);
fprintf(log_id,'Classes: %d\n',n_classes);
fprintf(log_id,'Epsilon: %d\n',epsilonw);
fprintf(log_id,'Weight Penalty: %d\n',weightcost);
fprintf(log_id,'Backprop Learning Rate: %d\n', SIG);

%%%% EXECUTION %%%%
makebatches;

pretrain;

backpropclassify_ES; 

save(times_file, 'time_rbm', 'time_backpropclassify');
fclose(log_id);
