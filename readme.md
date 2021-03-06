Convolutional Neural Network
================
 Introduction
 ------
 a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery.[1] They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics. They have applications in image and video recognition, recommender systems,image classification, medical image analysis, natural language processing,and financial time series. 
 A convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery.  
 CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "fully-connectedness" of these networks makes them prone to overfitting data. Typical ways of regularization include adding some form of magnitude measurement of weights to the loss function. CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme.

Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.

CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.
A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a ReLU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.
A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a ReLU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.[1]

Architecture of CNN
---------
In Regular Neural Nets, neural network mostly receive an input image having small size.
For example, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels) in regular neural networks. Full image cannot work well in regular  neural network.
So it is very important to use CNN to learning full images.
It is flexible to add many layers in the CNN. 
Usually, we can the following layers to build CNN:

(1) Convolutional layer: convolutional filter
(2) Activation layer: ReLu
(3) pooling layer: maxpooling
(4) Flatten layer
(5) Fully-connected layer
Convolutional layer, Activation layer,and pooling layer can be used more than once to extract more details from the page.
Besides, we can also use  soft-max layer and Dropout layer to avoid overfitting.

Process:
Data input ->( convolutional layer -> normalization layer -> max pooling layer)*n -> flatten layer -> fully connected layer -> output

### Key Points
• Build up a convolutional neural network consisting of Convolutional Networks, Batch Normalization, Max Pooling, and Backprop part using Python.
• Implement different scene recognition using convolutional neural network.
• Evaluate the testing results and optimize the framework of convolutional neural network.

Requirements
-----------
Python 3
&& Numpy
&& OpenCV

Contents
---------
## Basic architectures:
## Results:
## Evaluation:
  
  
  
#### Reference
###### (1)https://en.wikipedia.org/wiki/Convolutional_neural_network
###### (2)Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Communications of the ACM, 2017, 60(6): 84-90.
