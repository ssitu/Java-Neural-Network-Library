# Neural Network Class For Java

An adjustable neural network class, with an example of solving the XOR problem along with a classification variation.

Thought it would be easier to have the necessities of neural networks in one class. I add features when I need to use them in my other projects.

An instance of the neural network class can be instantiated like this:

NNest.NN neuralnetwork = new NNest().new NN(double learningRate, long seed, String hiddenActivationFunction, String outputActivationFunction, String costFunction, String optimizer, int ... layerNodes)

I could not think of another way to display a graph in JavaFX of the progress of the network, so I settled on a outer class that nests the neural network class inside.

Example 1: NNest.NN nn = new NNest().new NN(.001, 7777, "sigmoid", "sigmoid", "quadratic", "adam", 2, 2, 1);
This will make a neural network with 2 input nodes, 2 hidden nodes, and 1 output node.
The seed is set for repeatability.

Example 2: NNest.NN nn = new NNest().new NN(.001, 7777, "relu", "softmax", "log", "adam", 794, 512, 256, 128, 10); 
This will make a neural network with 794 input nodes, 512 hidden nodes in the first layer, 256 hidden nodes in the second, 128 hidden nodes in the third layer, and 10 output nodes.

To show a graph of the cost or accuracy can be shown with the static method NNest.graph(boolean, NN), where true will measure accuracy while false will measure cost. The NN argument takes in the neural network that will be used to plot the graph. There can only be one neural network that can be graphed currently, as there can not be more than one JavaFX thread.

NNest.graphJFX(boolean, NN) should be used when working with JavaFX applications. It will create a new stage instead of creating a new JavaFX thread to avoid the exception that comes with multiple JavaFX threads. The second argument takes in the neural network that will be graphed. This version of the method allows for graphs of different neural networks.

Names of available activation functions for the hidden layer: "sigmoid", "tanh", "relu", "leakyrelu"

Names of available activation functions for the output layer: regression: "sigmoid", "tanh", "linear"; classification: "softmax"

Names of available cost/loss/error functions: regression: "quadratic"; classification: "log"/"crossentropy"

Names of available optimizers/updaters: "" (for vanilla stochastic gradient descent), "momentum", "adam", "nadam"
