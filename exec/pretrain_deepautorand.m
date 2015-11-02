% Pretrain.m - Sorts out the pre-training

[numcases numdims numbatches]=size(batchdata);

n_distinct_layers = length(distinct_layer_nodes);
n_layers = length(layer_nodes);

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'This uses %3i epochs. \n',rbm_epoch);

maxepoch = rbm_epoch;

time_rbm = zeros(1, n_layers); % Initialize time variable

for i = 1:n_distinct_layers

   if i == 1
        tic; % Start timer
        fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',i,numdims,distinct_layer_nodes(i));
        restart=1;
        numhid = layer_nodes(i);
        maxepoch = rbm_epoch(i);
        rbmrand;
        hidrecbiases=hidbiases;
        pretrainweights{i}.vishid = vishid;
        pretrainweights{i}.hidbiases = hidbiases;
        pretrainweights{i}.visbiases = visbiases;
        time_rbm(1) = toc; % Stop timer
   else
        tic; % Start timer
        fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',i,layer_nodes(i-1),layer_nodes(i));
        batchdata=batchposhidprobs;
        restart=1;
        numhid = layer_nodes(i);
        maxepoch = rbm_epoch(i);
        rbmrand;     
        pretrainweights{i}.vishid = vishid;
        pretrainweights{i}.hidbiases = hidbiases;
        pretrainweights{i}.visbiases = visbiases;
        time_rbm(i) = toc;  % Stop timer
   end
    
save pretrain_weights pretrainweights
save(times_file, 'time_rbm');
end

