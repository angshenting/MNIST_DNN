% Version 1.3 
% 16 Sep 2013
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
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');

% Loading pretrained weights
load pretrain_weights

n_layers = length(layer_nodes);
n_distinctlayers = length(distinct_layer_nodes);
makebatches;
[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = cell(1, n_layers+1); % Pre-initialize cell
wprobs = cell(1, n_layers+1); % Pre-initialize cell
 %

%w{1}=[pretrainweights{1}.vishid; pretrainweights{1}.hidrecbiases];
for j = 1:n_distinctlayers
    w{j}=[pretrainweights{j}.vishid; pretrainweights{j}.hidbiases];
end

for jj = 1:n_distinctlayers
    w{n_layers+2-jj}=[pretrainweights{jj}.vishid';pretrainweights{jj}.visbiases];
end

clear j;
clear jj;

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l = zeros (n_layers+2,1); % Pre-initialize
for j = 1:n_layers+1
    l(j) = size(w{j},1)-1;
end
clear j;

l(n_layers+2)=l(1); 

test_err=[];
train_err=[];


for epoch = 1:maxepoch
tic; % Start timer
%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
%counter=0;
[numcases numdims numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches
  data = [batchdata(:,:,batch)];
  data = [data ones(N,1)];
  wprobs{1} = 1./(1+exp(-data*w{1})); wprobs{1} = [wprobs{1}  ones(N,1)];
  for j = 2:n_layers
      if j == n_distinctlayers
          wprobs{j} = wprobs{j-1}*w{j}; wprobs{j} = [wprobs{j} ones(N,1)];
      else
          wprobs{j} = 1./(1+exp(-wprobs{j-1}*w{j})); wprobs{j} = [wprobs{j} ones(N,1)];
      end
  end
  clear j;
  dataout = 1./(1 + exp(-wprobs{n_layers}*w{n_layers+1}));
  err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 )); 
  end
 train_err(epoch)=err/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% DISPLAY FIGURE TOP ROW REAL DATA BOTTOM ROW RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
%while 0 % Suppress Display
fprintf(1,'Displaying in figure 1: Top row - real data, Bottom row -- reconstructions \n');
output=[];
 for ii=1:15
  output = [output data(ii,1:end-1)' dataout(ii,:)'];
 end
   if epoch==1 
   close all 
   figure('Position',[100,600,1000,200]);
   else 
   figure(1)
   end 
   mnistdisp(output);
   drawnow;
%end % End Suppress Display
%%%%%%%%%%%%%%%%%%%% COMPUTE TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
counter=0;
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
err=0;
 for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  data = [data ones(N,1)];
  wprobs{1} = 1./(1+exp(-data*w{1})); wprobs{1} = [wprobs{1}  ones(N,1)];
  for j = 2:n_layers
      if j == n_distinctlayers
          wprobs{j} = wprobs{j-1}*w{j}; wprobs{j} = [wprobs{j} ones(N,1)];
      else
          wprobs{j} = 1./(1+exp(-wprobs{j-1}*w{j})); wprobs{j} = [wprobs{j} ones(N,1)];
      end
  end
  clear j;
  dataout = 1./(1 + exp(-wprobs{n_layers}*w{n_layers+1}));
  err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 )); 
 end
  
 test_err(epoch)=err/testnumbatches;
 fprintf(1,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \t \t \n',epoch,train_err(epoch),test_err(epoch));
 fprintf(log_id,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \t \t \n',epoch,train_err(epoch),test_err(epoch)); % Writes to log file
%%%%%%%%%%%%%% END OF COMPUTING TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/10
 fprintf(log_id,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;
  VV = [];
  
  for j = 1:n_layers+1
     VV = [VV transpose(w{j}(:))];
  end
  VV = VV';
  clear j;
    
  Dim = l;

  [X, fX] = minimize(VV,'CG_MNIST',log_id,max_iter,Dim,data,n_layers);

  xxx = 0;
  for j = 1:n_layers
        w{j} = reshape(X(xxx+1:xxx+(l(j)+1)*l(j+1)),l(j)+1,l(j+1));
        xxx = xxx+(l(j)+1)*l(j+1);
  end
  w{n_layers+1} = reshape(X(xxx+1:xxx+(l(n_layers+1)+1)*l(n_layers+2)),l(n_layers+1)+1,l(n_layers+2));

%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 save(weights_file, 'w');
 save(error_file, 'train_err', 'test_err');
 time_backprop(epoch) = toc; % Stop timer
 save(times_file,'time_rbm','time_backprop');

end



