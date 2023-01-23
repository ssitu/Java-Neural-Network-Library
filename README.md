# Neural Network Library For Java

An neural network library without dependencies outside of Java's built-in packages. It is coded with JDK 1.8 and it uses JavaFX that comes with JDK 1.8, but it is only used for tracking certain information of a neural network instance and is completely optional.

See the [test cases](https://github.com/ssitu/Neural-Network-Library-Class/tree/master/test/testcases) for small examples of using the library. 

# Features
## Layers
* Dense/Fully-connected
* Convolutional
* Flatten
* MaxPool
## Optimizers
* Momentum
* Nesterov (Dozat's version as a workaround)
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
