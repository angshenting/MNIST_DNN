% Version 1.3 
% 20 Aug 2013
% Shen Ting Ang
% 
% Latest changes: 
% - Updated version of Early Stopping; Refer to comments below for details
% - Added Validation Error calculations
%
% 
% Based on original code provided by Ruslan Salakhutdinov and Geoff Hinton
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

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in weights_file
% (controlled by main script)
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in Hinton's paper.  

maxepoch=backprop_epoch;
fprintf(1,'\nTraining discriminative model by minimizing cross entropy error. \n');


load pretrain_weights
n_layers = length(layer_nodes);
makebatches;
[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = cell(1, n_layers); % Pre-initialize cell

w{1}=[pretrainweights{1}.vishid; pretrainweights{1}.hidrecbiases];
for j = 2:n_layers
    w{j}=[pretrainweights{j}.hidpen; pretrainweights{j}.penrecbiases];
end
clear j; % Clear index
w_class = 0.03*randn(size(w{n_layers},2)+1,n_classes);

time_backpropclassify = zeros(1,maxepoch); % Initialize time variable

%%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l = zeros (n_layers+2,1); % Pre-initialize
for j = 1:n_layers
    l(j) = size(w{j},1)-1;
end
clear j;

l(n_layers+1)=size(w_class,1)-1;
l(n_layers+2)=n_classes; 

test_err=[];
train_err=[];


for epoch = 1:maxepoch
tic; % Start timer
%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
err_cr=0;
counter=0;
[numcases numdims numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches
  data = [batchdata(:,:,batch)];
  target = [batchtargets(:,:,batch)];
  data = [data ones(N,1)];
  %wprobs{1} = 1./(1 + exp(-data*w{1})); wprobs{1} = [wprobs{1}  ones(N,1)];
  wprobs{1} = data*w{1}; wprobs{1} = [wprobs{1}  ones(N,1)];
  for j = 2:n_layers
      wprobs{j} = 1./(1 + exp(-wprobs{j-1}*w{j})); wprobs{j} = [wprobs{j} ones(N,1)];
  end
  clear j;
  %wprobs{2} = 1./(1 + exp(-wprobs{1}*w{2})); wprobs{2} = [wprobs{2} ones(N,1)];
  %wprobs{3} = 1./(1 + exp(-wprobs{2}*w{3})); wprobs{3} = [wprobs{3}  ones(N,1)];
  targetout = exp(wprobs{n_layers}*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,n_classes);

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
 end
 train_err(epoch)=(numcases*numbatches-counter);
 train_crerr(epoch)=err_cr/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE VALIDATION MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0;
err_cr=0;
counter=0;
[validnumcases validnumdims validnumbatches]=size(validbatchdata);
N=validnumcases;


for batch = 1:validnumbatches
  data = [validbatchdata(:,:,batch)];
  target = [validbatchtargets(:,:,batch)];
  
  data = [data ones(N,1)];
  wprobs{1} = data*w{1}; wprobs{1} = [wprobs{1}  ones(N,1)];
  for j = 2:n_layers
      wprobs{j} = 1./(1 + exp(-wprobs{j-1}*w{j})); wprobs{j} = [wprobs{j} ones(N,1)];
  end
  clear j;
  targetout = exp(wprobs{n_layers}*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,n_classes);
  

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 
 valid_err(epoch)=(validnumcases*validnumbatches-counter);
 valid_crerr(epoch)=err_cr/validnumbatches;
 
%%%%%%%%%%%%%% END OF COMPUTING VALIDATION MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0;
err_cr=0;
counter=0;
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;


for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  target = [testbatchtargets(:,:,batch)];
  
  data = [data ones(N,1)];
  %wprobs{1} = 1./(1 + exp(-data*w{1})); wprobs{1} = [wprobs{1}  ones(N,1)];
  wprobs{1} = data*w{1}; wprobs{1} = [wprobs{1}  ones(N,1)];
  for j = 2:n_layers
      wprobs{j} = 1./(1 + exp(-wprobs{j-1}*w{j})); wprobs{j} = [wprobs{j} ones(N,1)];
  end
  clear j;
  targetout = exp(wprobs{n_layers}*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,n_classes);
  

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 
 
 test_err(epoch)=(testnumcases*testnumbatches-counter);
 test_crerr(epoch)=err_cr/testnumbatches;
  fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Validation # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches,valid_err(epoch),validnumcases*validnumbatches,test_err(epoch),testnumcases*testnumbatches);
 fprintf(log_id,'Before epoch %d Train # misclassified: %d (from %d). Validation # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches,valid_err(epoch),validnumcases*validnumbatches,test_err(epoch),testnumcases*testnumbatches);

%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% START OF EARLY STOPPING %%%%%%%%%%%%%%%%%%%%

% Reference: Prechelt (1998) - Early Stopping, but when?

% Criteria used: Either general loss (GL) (alpha = 1.03) or UP 5, i.e.
% first and third criteria

if epoch == 1 
    best_valid_err = valid_err(1);
else
   if valid_err(epoch) < best_valid_err % If validation error is lower
     best_valid_err = valid_err(epoch); % New best validation error
     
     % Save weights
     best_w = w;
     best_wprobs = wprobs;
     best_wclass = w_class;
     
     counter = 0;

   else
     % Calculate Generalization Loss
     general_loss = 100 * (valid_err(epoch)/best_valid_err - 1);
     counter = counter + 1;
     if general_loss > 101 || counter > 5
         w = best_w;
         w_class = best_wclass;
         save(weights_file, 'w', 'w_class');
         break;
     end
     
   end
 end

%%%%%%%%%%%%%% END OF EARLY STOPPING %%%%%%%%%%%%%%%%%%%%


 tt=0;
 for batch = 1:numbatches/10
 fprintf(log_id,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 targets=[]; 
 for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
  targets=[targets
        batchtargets(:,:,(tt-1)*10+kk)];
 end

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;
  
  if epoch<1 || n_layers == 1 % First update top-level weights holding other weights fixed. 
    N = size(data,1);
    XX = [data ones(N,1)];
    %wprobs{1} = 1./(1 + exp(-XX*w{1}));
    wprobs{1} = XX*w{1};
    if n_layers ~= 1
        wprobs{1} = [wprobs{1}  ones(N,1)];
    end
    for j = 2:n_layers
        wprobs{j} = 1./(1 + exp(-wprobs{j-1}*w{j})); 
        if j ~= n_layers
            wprobs{j} = [wprobs{j} ones(N,1)];
        end
    end
    clear j;
    %wprobs{3} = 1./(1 + exp(-wprobs{2}*w{3})); %wprobs{3} = [wprobs{3}  ones(N,1)];

    VV = [w_class(:)']';
    Dim = [l(n_layers+1); l(n_layers+2)];
    [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',SIG,log_id,max_iter,Dim,wprobs{n_layers},targets,n_classes);
    w_class = reshape(X,l(n_layers+1)+1,l(n_layers+2));

  else
    VV = [];
    for j = 1:n_layers
       VV = [VV transpose(w{j}(:))];
    end
    clear j;
    VV = [VV transpose(w_class(:))]';
    Dim = l;
    [X, fX] = minimize(VV,'CG_CLASSIFY',SIG,log_id,max_iter,Dim,data,targets,n_layers,n_classes);

    
    w{1} = reshape(X(1:(l(1)+1)*l(2)),l(1)+1,l(2));
    xxx = (l(1)+1)*l(2);
    for j = 2:n_layers
        w{j} = reshape(X(xxx+1:xxx+(l(j)+1)*l(j+1)),l(j)+1,l(j+1));
        xxx = xxx+(l(j)+1)*l(j+1);
    end
    
    w_class = reshape(X(xxx+1:xxx+(l(n_layers+1)+1)*l(n_layers+2)),l(n_layers+1)+1,l(n_layers+2));

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 save(weights_file, 'w', 'w_class');
 save(error_file, 'train_err', 'train_crerr', 'valid_err', 'valid_crerr', 'test_err', 'test_crerr');
 time_backpropclassify(epoch) = toc; % Stop timer
 save(times_file,'time_rbm','time_backpropclassify');
end

