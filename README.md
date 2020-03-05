# NeuralNetwork
An adjustable neural network class

Thought it would be easier to have the necessities of neural networks in one class.

An instance of the neural network class can be instantiated like this:

NNest.NN neuralnetwork = new NNest().new NN(double learningRate, long seed, String hiddenActivationFunction, String outputActivationFunction, String costFunction, String optimizer, int ... layerNodes)

Example 1: NNest.NN nn = new NNest().new NN(.001,7777,"sigmoid","sigmoid","quadratic","adam",2,2,1);
This will make a neural network with 2 input nodes, 2 hidden nodes, and 1 output nodes.
The seed is set for repeatability.

Example 2: NNest.NN nn = new NNest().new NN(.001,7777,"relu","softmax","log","adam",794,512,256,128,10); 
This will make a neural network with 794 input nodes, 512 hidden nodes in the first layer, 256 hidden nodes in the second, 128 hidden nodes in the third layer, and 10 output nodes.

To show a graph of the cost or accuracy can be shown with the static method NNest.graph(boolean), where true will measure accuracy while false will measure cost.
NNest.graphJFX(boolean) can be used when working with JavaFX applications since there cannot be multiple JavaFX threads and it must be launched from the JavaFX thread.

Names of available activation functions for the hidden layer: sigmoid, tanh, relu, leakyrelu

Names of available activation functions for the output layer: regression: sigmoid, tanh, linear; classification: softmax

Names of available cost/loss/error functions: regression: quadratic; classification: log

Names of available optimizers/updaters: momentum, adam
