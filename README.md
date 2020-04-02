# Neural Network Class/Library For Java

An adjustable neural network class or library, with an example of solving the XOR problem along with a classification variation.

Thought it would be easier to have the necessities of neural networks in one class. I add features when I need to use them in my other projects.

I could not think of another way to display a graph in JavaFX of the progress of the network, so I settled on a outer class that nests the neural network class inside.

See the XOR example for an example on initializing a neural network: https://github.com/SSithub/Neural-Network-Library-Class/blob/Floats/src/neuralnetwork/XOR.java

To show a graph of the cost or accuracy can be shown with the static method NNLib.graph(boolean, NN), where true will measure accuracy while false will measure cost. The NN argument takes in the neural network that will be used to plot the graph. There can only be one neural network that can be graphed currently, as there can not be more than one JavaFX thread. Attempting to call this method more than once will cause a JavaFX exception but is handled by JavaFX. The new call will cause the graph to plot the NN that is passed, replacing the old NN.

NNLib.graphJFX(boolean, NN) should be used when working with JavaFX applications. It will create a new stage instead of creating a new JavaFX thread to avoid the exception that comes with multiple JavaFX threads. The second argument takes in the neural network that will be graphed. This version of the method allows for graphs of different neural networks.

The parameters take in enums from the NNLib class:

    public enum Initializer {
        VANILLA, XAVIER, HE
    }

    public enum ActivationFunction {
        LINEAR, SIGMOID, TANH, RELU, LEAKYRELU, SWISH, MISH, CUSTOM,
        SOFTMAX
    }

    public enum LossFunction {
        QUADRATIC(.5), HUBER(1), HUBERPSEUDO(1), CUSTOM(0),
        CROSS_ENTROPY(0);

        private float steepness;

        private LossFunction(double steepnessFactor) {
            steepness = (float) steepnessFactor;
        }

        public LossFunction steepness(double steepness) {
            this.steepness = (float) steepness;
            return this;
        }

    }

    public enum Optimizer {
        VANILLA, MOMENTUM, RMSPROP, ADAM, ADAMAX, NADAM, AMSGRAD
    }
