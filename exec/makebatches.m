% Version 2.000
%
% Shen Ting Ang, 16 April 2013
%
% Based on code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own
% risk.

%%%%%%%%%%%%%%%% TRAINING DATA %%%%%%%%%%%%%%%%%%%%%%%%
traindata=[]; 
targets=[]; 

load sortedtraindata;

for i = 1:n_classes
    
    traindata = [traindata; sortedtrain{i}];
    
    targetvec = zeros(1,n_classes);
    targetvec(i) = 1;
    targets = [targets;repmat(targetvec, size(sortedtrain{i},1), 1)];
    
    % Normalisation
    
end
traindata = traindata/255;

totnum=size(traindata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

%rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(traindata,2);
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, n_classes, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = traindata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear traindata targets;

%%%%%%%%%%%%%%%% VALIDATION DATA %%%%%%%%%%%%%%%%%%%%%%%%

validdata=[]; 
targets=[]; 

load sortedvaliddata;

for i = 1:n_classes
    
    validdata = [validdata; sortedvalid{i}];
    
    targetvec = zeros(1,n_classes);
    targetvec(i) = 1;
    targets = [targets;repmat(targetvec, size(sortedvalid{i},1), 1)];
    
    % Normalisation
    
end
validdata = validdata/255;

totnum=size(validdata,1);
fprintf(1, 'Size of the validation dataset= %5d \n', totnum);

randomorder=randperm(totnum);

numbatches=totnum/batchsize;
numdims  =  size(validdata,2);
validbatchdata = zeros(batchsize, numdims, numbatches);
validbatchtargets = zeros(batchsize, n_classes, numbatches);

for b=1:numbatches
  validbatchdata(:,:,b) = validdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  validbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear validdata targets;

%%%%%%%%%%%%%%%% TESTING DATA %%%%%%%%%%%%%%%%%%%%%%%%
testdata=[]; 
targets=[]; 

load sortedtestdata;

for i = 1:n_classes
    
    testdata = [testdata; sortedtest{i}];
    
    targetvec = zeros(1,n_classes);
    targetvec(i) = 1;
    targets = [targets;repmat(targetvec, size(sortedtest{i},1), 1)];
    
    % Normalisation
    
end
testdata = testdata/255;

totnum=size(testdata,1);
fprintf(1, 'Size of the testing dataset= %5d \n', totnum);

originalseed = rng;
rng(1);
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(testdata,2);
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, n_classes, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = testdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear testdata targets;
%%% Reset random seeds 
rng(originalseed);