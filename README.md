# BayDNN-w-CD
This repository consists of VGG16 network with batch normalization. The network was trained on 
CIFAR-10 dataset using the network configuration (specifically Dropout) described here:
http://torch.ch/blog/2015/07/30/cifar.html

This pretrained network is then fine tuned using Concrete Dropout described in the following paper:
http://papers.neurips.cc/paper/6949-concrete-dropout.pdf