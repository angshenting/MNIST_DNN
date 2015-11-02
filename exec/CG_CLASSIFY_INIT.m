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


function [f, df] = CG_CLASSIFY_INIT(VV,Dim,wprobs_top,target,n_classes);
l1 = Dim(1);
l2 = Dim(2);

N = size(wprobs_top,1);
% Do decomversion.
  w_class = reshape(VV,l1+1,l2);
  wprobs_top = [wprobs_top ones(N,1)];  
  %wprobs_top = [wprobs_top];  

  targetout = exp(wprobs_top*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,n_classes);
  f = -sum(sum( target(:,1:end).*log(targetout))) ;
IO = (targetout-target(:,1:end));
Ix_class=IO; 
dw_class =  wprobs_top'*Ix_class; 

df = [dw_class(:)']'; 

