load mnist_classify_weights

testdata = csvread('test.csv',1,0);
testlength = size(testdata,1);

testdata = [testdata ones(testlength,1)];

w1probs = 1./(1 + exp(-testdata*w1)); w1probs = [w1probs  ones(testlength,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(testlength,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(testlength,1)];
w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(testlength,1)];

testout = exp(w4probs*w_class);
  
targetout = testout./repmat(sum(testout,2),1,10);
[I J]=max(targetout,[],2);
J = J-1;
csvwrite('classify_output.csv',J);