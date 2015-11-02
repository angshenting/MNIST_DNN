% Pretrain.m - Sorts out the RBM pre-training. 

[numcases numdims numbatches]=size(batchdata);

n_distinct_layers = length(distinct_layer_nodes);

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'This uses %3i epochs. \n',rbm_epoch);

maxepoch = rbm_epoch;

time_rbm = zeros(1, n_distinct_layers); % Initialize time variable

for i = 1:n_distinct_layers
    
    tic;
    fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n',i,numdims,distinct_layer_nodes(i));
    restart=1;
    numhid = layer_nodes(i);
    maxepoch = rbm_epoch(i);
    if i == 1
        switch layer_type(i);
            case 'sigmoid'
                rbm;
            case 'gaussian'
                rbmvislinear;
            case 'linear'
                rbmhidlinear;
            otherwise
                fprintf(1,'Type unknown\n');
                break;
        end
        
        hidrecbiases=hidbiases;
        pretrainweights{i}.vishid = vishid;
        pretrainweights{i}.hidbiases = hidbiases;
        pretrainweights{i}.visbiases = visbiases;
        time_rbm(1) = toc; % Stop timer
    else
         switch layer_type(i);
             case 'sigmoid'
                rbm;
            case 'gaussian'
                rbmvislinear;
            case 'linear'
                rbmhidlinear;
            otherwise
                fprintf(1,'Type unknown\n');
                break;
         end
         pretrainweights{i}.vishid = vishid;
         pretrainweights{i}.hidbiases = hidbiases;
         pretrainweights{i}.visbiases = visbiases;
    end
   
    
save pretrain_weights pretrainweights
save(times_file, 'time_rbm');
end

