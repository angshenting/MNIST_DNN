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

% This program reads raw MNIST files available at 
% http://yann.lecun.com/exdb/mnist/ 
% and converts them to files in matlab format 
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.  

% This program was originally written by Yee Whye Teh 

n_classes = 10;

% Work with test files first 
fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n'); 

f = fopen('t10k-images-idx3-ubyte','r');
[a,count] = fread(f,4,'int32');
  
g = fopen('t10k-labels-idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Testing MNIST images (prints 60 dots)\n'); 
n = 1000;

sortedtest = cell(1,n_classes);

for i=1:10,
  fprintf('.');
  rawimages = fread(f,28*28*n,'uchar');
  rawlabels = fread(g,n,'uchar');
  rawimages = reshape(rawimages,28*28,n);
  rawimages = rawimages/255;

  for j=1:n,
    sortedtest{rawlabels(j)+1} = [sortedtest{rawlabels(j)+1}; rawimages(:,j)'];
    
  end;
end;
 fprintf('\n');
save sortedtestdata.mat sortedtest;


% Work with training files second  
f = fopen('train-images-idx3-ubyte','r');
[a,count] = fread(f,4,'int32');

g = fopen('train-labels-idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n'); 
n = 1000;

sortedtrain = cell(1,n_classes);
sortedvalid = cell(1,n_classes);

for i=1:60,
  fprintf('.');
  rawimages = fread(f,28*28*n,'uchar');
  rawlabels = fread(g,n,'uchar');
  rawimages = reshape(rawimages,28*28,n);
  rawimages = rawimages/255;
  
  if i <=50
    for j=1:n,
        sortedtrain{rawlabels(j)+1} = [sortedtrain{rawlabels(j)+1}; rawimages(:,j)'];
    end;
  else
      for j=1:n,
        sortedvalid{rawlabels(j)+1} = [sortedvalid{rawlabels(j)+1}; rawimages(:,j)'];
      end;
  end
end;
 fprintf('\n');
save sortedtraindata.mat sortedtrain;
save sortedvaliddata.mat sortedvalid;
