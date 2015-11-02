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

function [f, df] = CG_MNIST(VV,Dim,XX,n_layers);

N = size(XX,1);

% Decomversion
xxx = 0;
for j = 1:n_layers+1
    start_idx = xxx+1;
    end_idx = xxx+(Dim(j)+1)*Dim(j+1);
    w{j} = reshape(VV(start_idx:end_idx),Dim(j)+1,Dim(j+1));
    xxx = xxx+(Dim(j)+1)*Dim(j+1);
end
clear j;

XX = [XX ones(N,1)];
wprobs{1} = sigmoid(XX,w{1}); wprobs{1} = [wprobs{1} ones(N,1)];
for j = 2:n_layers
    if j == (n_layers+1)/2
        wprobs{j} = wprobs{j-1}*w{j};
    else
        wprobs{j} = sigmoid(wprobs{j-1},w{j});
    end
    wprobs{j} = [wprobs{j} ones(N,1)];
end
clear j;

XXout = sigmoid(wprobs{n_layers},w{n_layers+1});

f = -1/N*sum(sum( XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout)));
IO = 1/N*(XXout-XX(:,1:end-1));

Ix = cell(1,n_layers+1);
dw = cell(1,n_layers+1);

Ix{n_layers+1}=IO; 
dw{n_layers+1} =  transpose(wprobs{n_layers})*Ix{n_layers+1}; 


for k = 1:n_layers
	idx = n_layers+1-k;
    if k == n_layers % For Ix1
        Ix{idx} = (Ix{idx+1}*transpose(w{idx+1})).*wprobs{idx}.*(1-wprobs{idx});
		Ix{idx} = Ix{idx}(:,1:end-1);
		dw{idx} = XX'*Ix{idx};
    elseif k == (n_layers+1)/2
		Ix{idx} = (Ix{idx+1}*transpose(w{idx+1}));%.*wprobs{idx}.*(1-wprobs{idx});
		Ix{idx} = Ix{idx}(:,1:end-1);
		dw{idx} = transpose(wprobs{idx-1})*Ix{idx};
    else
        Ix{idx} = (Ix{idx+1}*transpose(w{idx+1})).*wprobs{idx}.*(1-wprobs{idx});
		Ix{idx} = Ix{idx}(:,1:end-1);
		dw{idx} = transpose(wprobs{idx-1})*Ix{idx};
    end
end
clear k;
clear idx;

df=[];
for k = 1:n_layers+1
	df = [df transpose(dw{k}(:))];
end
df = df';