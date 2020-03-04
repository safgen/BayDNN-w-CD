# BayDNN-w-CD
This repository consists of VGG16 network with batch normalization. The network was trained on 
CIFAR-10 dataset using the network configuration (specifically Dropout) described here:
http://torch.ch/blog/2015/07/30/cifar.html

This pretrained network is then fine tuned using Concrete Dropout described in the following paper:
http://papers.neurips.cc/paper/6949-concrete-dropout.pdf

LATEST ACCURACIES:

Training accuracy after pretraining: 0.9164199829101562

Test accuracy after pretraining: 0.8870000243186951

Train accuracy after fine-tuning with concrete dropout: 0.9259999990463257, 

Test accuracy after fine-tuning with concrete dropout: 0.9019000029563904