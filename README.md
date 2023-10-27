# Neural Network Library For Java

An neural network library without dependencies outside of Java's built-in packages. This was a project for learning both Java and neural networks. It was the largest project I've attempted at the time, so the code and design choices are not great. It uses JDK 1.8 and uses JavaFX that comes with JDK 1.8, but it is only used for data visualization in [JavaFXTools.java](src/nnlibrary/JavaFXTools.java) so the dependency can be easily removed.

See the [test cases](test/testcases) for small examples of using the library. 

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
## Customizable
* All of the above categories can be extended to create new modules. An example can be found here: [XOR_Classification.java](test/testcases/XOR_Classification.java#L15-L34)
## General Features
* Built-in saving and loading of network parameters with both uncompiled and compiled .jar support
* Displaying information about a neural network instance with JavaFX
    * A scrollable window with all parameters of each layer in the network
    * A graph that measures loss or accuracy of the network over each iteration of backpropagation
