
protein map function by CNN
======

Overview

To make the protein map function which can judge whether protein map or not,
the function is made by using convolution neural network.

First, the resolution between 10 to 4.5 angstrom, and the space group P212121
are select to learn.


The 20180808b files are the results of the model and weights by learning of FC(all pdb) and Fo(all strcuture data) witout 4rfu data. 

the feature of function around the correct solution was investigated 4rfu
by ../ml_fo_test.py. The result plot is "4rfuplot.jpg".
y-axis is the value of function x-axis is phase difference as sigma of gaussian in degree.



===========
Requirement

Python3

Tensorflow keras

ccp4

numpy

PDB data (gziped)

===========

