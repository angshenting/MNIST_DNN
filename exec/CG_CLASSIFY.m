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


function [f, df] = CG_CLASSIFY(VV,Dim,XX,target,n_layers,n_classes);


N = size(XX,1);
XX = [XX ones(N,1)];

% Do decomversion.
w = cell(1,n_layers);
wprobs = cell(1,n_layers);
w{1} = reshape(VV(1:(Dim(1)+1)*Dim(2)),Dim(1)+1,Dim(2));
		xxx = (Dim(1)+1)*Dim(2);
		wprobs{1} = XX*w{1}; 
		wprobs{1} = [wprobs{1} ones(N,1)];

for k = 2:n_layers
	%if k == 1
	%	w{k} = reshape(VV(1:(Dim(k)+1)*Dim(k+1)),Dim(k)+1,Dim(k+1));
	%	xxx = (Dim(k)+1)*Dim(k+1);
	%	wprobs{k} = 1./(1 + exp(-XX*w{k})); 
	%	wprobs{k} = [wprobs{k}  ones(N,1)];
	%else
		w{k} = reshape(VV(xxx+1:xxx+(Dim(k)+1)*Dim(k+1)),Dim(k)+1,Dim(k+1));
		xxx=xxx+(Dim(k)+1)*Dim(k+1);
		wprobs{k} = 1./(1 + exp(-wprobs{k-1}*w{k})); 
		wprobs{k} = [wprobs{k} ones(N,1)];
	%end
end
clear k;

w_class = reshape(VV(xxx+1:xxx+(Dim(n_layers+1)+1)*Dim(n_layers+2)),Dim(n_layers+1)+1,Dim(n_layers+2));

  targetout = exp(wprobs{n_layers}*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,n_classes);
  f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class=IO; 
dw_class =  transpose(wprobs{n_layers})*Ix_class; 

Ix = cell(1,n_layers);
dw = cell(1,n_layers);
for k = 1:n_layers
	idx = n_layers+1-k;
    if k == n_layers
        Ix{idx} = (Ix{idx+1}*transpose(w{idx+1}));%.*wprobs{idx}.*(1-wprobs{idx});
		Ix{idx} = Ix{idx}(:,1:end-1);
		dw{idx} = XX'*Ix{idx};
    elseif k == 1
		Ix{idx} = (Ix_class*w_class').*wprobs{idx}.*(1-wprobs{idx});
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
for k = 1:n_layers
	df = [df transpose(dw{k}(:))];
end

df = [df dw_class(:)']';
