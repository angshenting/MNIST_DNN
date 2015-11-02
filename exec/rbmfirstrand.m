% Pretrain.m - Initializes the pre-training, originally part of
% mnistclassify/mnistautoencode.m

makebatches;
[numcases numdims numbatches]=size(batchdata);

n_layers = length(layer_nodes);

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n',rbm_epoch);



time_rbm = zeros(1, n_layers); % Initialize time variable
%makebatches2;
%[numcases numdims numbatches]=size(batchdata);

for i = 1:n_layers

   if i == 1
        tic; % Start timer
        fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',i,numdims,layer_nodes(i));
        restart=1;
        numhid = layer_nodes(i);
        maxepoch = rbm_epoch(i);
        rbmvislinear;
        hidrecbiases=hidbiases;
        %filename = 'vhclassify';
        %save(filename,'vishid', 'hidrecbiases', 'visbiases');
        pretrainweights{i}.vishid = vishid;
        pretrainweights{i}.hidrecbiases = hidbiases;
        pretrainweights{i}.visbiases = visbiases;
        time_rbm(1) = toc; % Stop timer
   else
        tic; % Start timer
        fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',i,layer_nodes(i-1),layer_nodes(i));
        batchdata=batchposhidprobs;
        restart=1;
        numhid = layer_nodes(i);
        maxepoch = rbm_epoch(i);
        rbm;
        
        %eval(['hidpen' i-1 '=vishid']); 
        %eval(['penrecbiases' i-1 '=hidbiases']); 
        %eval(['hidgenbiases' i-1 '=visbiases']);
        %filename = ['hpclassify' num2str(i-1)];
        %save(filename,'vishid','hidrecbiases','visbiases'); 
        
        pretrainweights{i}.hidpen = vishid;
        pretrainweights{i}.penrecbiases = hidbiases;
        pretrainweights{i}.hidgenbiases = visbiases;
        time_rbm(i) = toc;  % Stop timer
   end
    
save pretrain_weights pretrainweights


save(times_file, 'time_rbm');
end

