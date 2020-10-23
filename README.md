# Neural Network Class/Library For Java

An neural network class/library without dependencies outside of Java's API packages. It is coded with JDK 1.8 and it does use JavaFX that comes with the 1.8 version, but it is only used for tracking certain information of a neural network instance and is completely optional. I decided to use floats instead of doubles to trade precision for speed, but it can easily use doubles by replacing all occurances of float and Float with double and Double. The difference in performance is significant.

Thought it would be easier to have the necessities of neural networks in one class, [the NNlib class](https://github.com/ssitu/Neural-Network-Library-Class/blob/master/src/nnlibrary/NNlib.java). I add features when I need to use them in my other projects.

See the [test cases](https://github.com/ssitu/Neural-Network-Library-Class/tree/master/test/testcases) for small examples of using the class. 

# Features
## Layers
* Dense/Fully-connected
* Convolutional
* Flatten
* MaxPool
## Optimizers
* Momentum
* Nesterov (Dozat's rewrite)
* Adagrad
* Adadelta
* RMSProp
* Adam
* Adamax
* Nadam
* AMSGrad
## Loss Functions
* Quadratic
* Huber
* Pseudo Huber
* Cross-entropy
## Activation Functions
* Linear
* Sigmoid
* Tanh
* ReLU
* Leaky ReLU
* Swish
* Mish
* Softmax (Sometimes acts weird, not sure why)
## Weight Initializers
* Xavier
* He
## Other
* Min Max Normalization
* Z Score Normalization
* Tanh Estimator Normalization
## General Features
* Built-in saving and loading of network parameters with both uncompiled and .jar support
* Displaying information about a neural network instance with JavaFX
    * A scrollable window with all parameters of each layer in the network
    * A graph that measures loss or accuracy of the network over each iteration of backpropagation
