MNIST_DNN - An easily customizable and transparent basic deep learning system in MATLAB

Based on code originally written by Geoff Hinton et al. See http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
Additional features are the ability to customize the network parameters such as the number of nodes per layer and the type of activation units more easily. The code currently only supports sigmoid, gaussian and linear units.

How to use:
- Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and put the files in the data subfolder
- Run converter2.m (you will only need to run this once) 
- Run mnistdeepauto.m or mnistclassify.m

Code was written in 2013. This is purely for archival purposes. I am not actively maintaining this repository. If you are looking for a production-grade package, I strongly suggest using Python with Theano and Keras - trust me, it is a lot less painful.