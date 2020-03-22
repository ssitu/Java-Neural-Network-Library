package neuralnetwork;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class NNest extends Application implements Serializable {

    volatile static double globalCost;
    volatile static int sessions = 0;
    private static boolean graphMeasuresAccuracy;

    public class NN implements Serializable {

        private class Layer implements Serializable {

            float[][] weights;
            float[][] biases;

            Layer(int previousNodes, int nodes) {
                weights = create(previousNodes, nodes, 0);
                biases = create(1, nodes, 0);
                weights = scale(randomize(weights, 2, -1), (float) Math.sqrt(2.0 / previousNodes));
                biases = randomize(biases, 2, -1);
            }
        }
        private int threads;
        private String name;
        private Layer[] network;
        private Random random = new Random();
        private long seed;
        private float lr;
        private double cost;
        final int NETWORKSIZE;//Total layers not including the input layer
        private String hiddenActivationFunction;
        private String outputActivationFunction;
        private String costFunction;
        private String optimizer;
        private int[] layerNodes;
        private float[][][][] previousMomentsW;
        private float[][][][] previousMomentsB;
        private transient BiFunction<float[][], Boolean, float[][]> activationHiddens;
        private transient BiFunction<float[][], Boolean, float[][]> activationOutputs;
        private transient BiFunction<float[][], float[][], Function<Boolean, float[][]>> lossFunction;
        private transient BiFunction<Float, float[][], BiFunction<float[][], float[][], float[][][]>> updater;
        private transient BiFunction<float[][], float[][], float[][]> dotProduct = (a, b) -> dot(a, b);

        /**
         * @param saveName Used in the name of the save file
         * @param learningRate Learning rate for the gradient descent.
         * @param seed Seed for reproducible random values.
         * @param hiddenActivationFunction Activation function for the hidden
         * layers (sigmoid, tanh, relu, leakyrelu).
         * @param outputActivationFunction Activation function for the output
         * layer (regression: sigmoid, tanh, linear; classification: softmax).
         * @param costFunction Cost function to measure error (regression:
         * quadratic; classification: log).
         * @param optimizer Gradient updater for stochastic gradient descent,
         * leave blank for none (momentum, adam, nadam).
         * @param layerNodes Amount of numbers specifies the amount of layers
         * while the value of the numbers specifies the amount of neurons for
         * that layer. Must have more than two numbers (input layer, hidden
         * layers, output layer).
         */
        NN(String saveName, double learningRate, long seed, String hiddenActivationFunction, String outputActivationFunction, String costFunction, String optimizer, int... layerNodes) {
            name = saveName;
            lr = (float) learningRate;
            random.setSeed(seed);
            this.hiddenActivationFunction = hiddenActivationFunction;
            this.outputActivationFunction = outputActivationFunction;
            this.costFunction = costFunction;
            this.optimizer = optimizer;
            this.layerNodes = layerNodes;
            network = new Layer[layerNodes.length - 1];
            if (layerNodes.length < 3) {
                throw new IllegalArgumentException("MUST HAVE MORE THAN 2 LAYERS IN THE NEURAL NETWORK");
            }
            //Activation functions for the hidden layers
            if ("sigmoid".equalsIgnoreCase(hiddenActivationFunction)) {
                activationHiddens = (a, b) -> sigmoidActivation(a, b);
            } else if ("tanh".equalsIgnoreCase(hiddenActivationFunction)) {
                activationHiddens = (a, b) -> tanhActivation(a, b);
            } else if ("relu".equalsIgnoreCase(hiddenActivationFunction)) {
                activationHiddens = (a, b) -> reluActivation(a, b);
            } else if ("leakyrelu".equalsIgnoreCase(hiddenActivationFunction)) {
                activationHiddens = (a, b) -> leakyReluActivation(a, b);
            } else {
                throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE HIDDEN LAYERS");
            }
            //Activation functions for the output layer
            if ("sigmoid".equalsIgnoreCase(outputActivationFunction)) {
                activationOutputs = (a, b) -> sigmoidActivation(a, b);
            } else if ("tanh".equalsIgnoreCase(outputActivationFunction)) {
                activationOutputs = (a, b) -> tanhActivation(a, b);
            } else if ("softmax".equalsIgnoreCase(outputActivationFunction)) {
                activationOutputs = (a, b) -> softmaxActivation(a, b);
            } else if ("linear".equalsIgnoreCase(outputActivationFunction)) {
                activationOutputs = (a, b) -> linearActivation(a, b);
            } else {
                throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE OUTPUT LAYER");
            }
            //Cost functions for the backpropagation
            if ("quadratic".equalsIgnoreCase(costFunction)) {
                this.lossFunction = (a, b) -> (c) -> quadraticLoss(a, b, c);
            } else if ("log".equalsIgnoreCase(costFunction))//The target should be passed as making the correct classification as the highest value
            {
                this.lossFunction = (a, b) -> (c) -> logLoss(a, b, c);
            } else {
                throw new IllegalArgumentException("INVALID COST FUNCTION");
            }
            //Optimizer
            setOptimizer(optimizer);
            //Adding each layer
            for (int i = 1; i < layerNodes.length; i++) {
                Layer layer = new Layer(layerNodes[i - 1], layerNodes[i]);
                network[i - 1] = layer;
            }
            NETWORKSIZE = network.length;
            previousMomentsW = new float[2][NETWORKSIZE][][];
            previousMomentsB = new float[2][NETWORKSIZE][][];
            for (int i = 0; i < layerNodes.length - 1; i++) {
                int rows = network[i].weights.length;
                int columns = network[i].weights[0].length;
                previousMomentsW[0][i] = create(rows, columns, 0);
                previousMomentsB[0][i] = create(1, columns, 0);
                previousMomentsW[1][i] = create(rows, columns, 0);
                previousMomentsB[1][i] = create(1, columns, 0);
            }
        }

        @Override
        public String toString() {
            String networkLayers = "";
            for (Layer layer : network) {
                networkLayers += layer.weights.length + ",";
            }
            networkLayers += network[NETWORKSIZE - 1].weights[0].length;
            return networkLayers;
        }

        public Layer getNetworkLayer(int layerIndex) {//Layer 0 is the layer after the inputs (first hidden layer)
            try {
                return network[layerIndex];
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }

        public int getNetworkSize() {
            return network.length;
        }

        public NN clone() {
            NN nnCopy = new NN(name, lr, seed, hiddenActivationFunction, outputActivationFunction, costFunction, optimizer, layerNodes);
            for (int i = 0; i < getNetworkSize(); i++) {
                nnCopy.getNetworkLayer(i).weights = copy(getNetworkLayer(i).weights);
                nnCopy.getNetworkLayer(i).biases = copy(getNetworkLayer(i).biases);
            }
            return nnCopy;
        }

        public void save() {
            try {
                FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + "/" + name + "_neuralnetwork(" + toString() + ")");
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                out.writeObject(this);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public boolean load() {
            try {
                FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + "/" + name + "_neuralnetwork(" + toString() + ")");
                ObjectInputStream in = new ObjectInputStream(fileIn);
                network = ((NN) in.readObject()).network;
//                random = ((NN)in.readObject()).random;
                return true;
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("Could not load network settings.");
                e.printStackTrace();
                return false;
            }
        }

        public void mutateAdditive(double mutateRate, double range) {
            for (int i = 0; i < getNetworkSize(); i++) {
                for (int j = 0; j < getNetworkLayer(i).weights.length; j++) {
                    for (int k = 0; k < getNetworkLayer(i).weights[0].length; k++) {
                        if (Math.random() < mutateRate) {
                            getNetworkLayer(i).weights[j][k] += (float) (Math.random() * range - (range / 2));
                        }
                    }
                }
                for (int j = 0; j < getNetworkLayer(i).biases.length; j++) {
                    for (int k = 0; k < getNetworkLayer(i).biases[0].length; k++) {
                        if (Math.random() < mutateRate) {
                            getNetworkLayer(i).biases[j][k] += (float) (Math.random() * range - (range / 2));
                        }
                    }
                }
            }
        }

        public void mutateNewValues(double mutateRate, double range) {
            for (int i = 0; i < getNetworkSize(); i++) {
                for (int j = 0; j < getNetworkLayer(i).weights.length; j++) {
                    for (int k = 0; k < getNetworkLayer(i).weights[0].length; k++) {
                        if (Math.random() < mutateRate) {
                            getNetworkLayer(i).weights[j][k] = (float) (Math.random() * range - (range / 2));
                        }
                    }
                }
                for (int j = 0; j < getNetworkLayer(i).biases.length; j++) {
                    for (int k = 0; k < getNetworkLayer(i).biases[0].length; k++) {
                        if (Math.random() < mutateRate) {
                            getNetworkLayer(i).biases[j][k] = (float) (random.nextFloat() * range - (range / 2));
                        }
                    }
                }
            }
        }

        public void randomizeNetwork(double range) {
            for (int i = 0; i < getNetworkSize(); i++) {
                for (int j = 0; j < getNetworkLayer(i).weights.length; j++) {
                    for (int k = 0; k < getNetworkLayer(i).weights[0].length; k++) {
                        getNetworkLayer(i).weights[j][k] = (float) (random.nextFloat() * range - (range / 2));
                    }
                }
                for (int j = 0; j < getNetworkLayer(i).biases.length; j++) {
                    for (int k = 0; k < getNetworkLayer(i).biases[0].length; k++) {
                        getNetworkLayer(i).biases[j][k] = (float) (random.nextFloat() * range - (range / 2));
                    }
                }
            }
        }

        public float[][] feedforward(float[][] inputs) {
            float[][] outputs = inputs;
            for (int i = 0; i < NETWORKSIZE - 1; i++) {//Feed the inputs through the hidden layers
                Layer currentLayer = network[i];
                outputs = activationHiddens.apply(add(dotProduct.apply(outputs, currentLayer.weights), currentLayer.biases), false);
            }
            //Feed the output from the hidden layers to the output layers with its activation function
            Layer lastLayer = network[NETWORKSIZE - 1];
            outputs = activationOutputs.apply(add(dotProduct.apply(outputs, lastLayer.weights), lastLayer.biases), false);
            return outputs;
        }

        public void backpropagation(float[][] inputs, float[][] targets) {//Using notation from neuralnetworksanddeeplearning.com
            Layer lastLayer = network[NETWORKSIZE - 1];
            if (targets[0].length != lastLayer.biases[0].length) {
                throw new IllegalArgumentException("TARGETS ARRAY DO NOT MATCH THE SIZE OF THE OUTPUT LAYER");
            }
            float[][] outputs = inputs;
            //Each partial derivative is used in this order
            float[][] dC_dA;
            float[][] dA_dZ;
            float[][] dZ_dW;
            float[][] dC_dZ = {{}};
            float[][] dC_dW;
            float[][] bGradients;
            float[][] wGradients;
            float[][] dZ_dA = {{}};
            float[][][] Z = new float[NETWORKSIZE][][];//"Z" = the unactivated inputs from the weights and biases
            float[][][] A = new float[NETWORKSIZE + 1][][];//"A" = the activated "Z"s
            A[0] = outputs;
            for (int i = 0; i < NETWORKSIZE - 1; i++) {
                Layer currentLayer = network[i];//Increase performance by reducing amount of pointers
                outputs = add(dotProduct.apply(outputs, currentLayer.weights), currentLayer.biases);//Computing "Z"
                Z[i] = outputs;
                outputs = activationHiddens.apply(outputs, false);//Computing "A"
                A[i + 1] = outputs;
            }
            outputs = add(dotProduct.apply(outputs, lastLayer.weights), lastLayer.biases);
            Z[NETWORKSIZE - 1] = outputs;
            outputs = activationOutputs.apply(outputs, false);
            A[NETWORKSIZE] = outputs;
            dC_dA = lossFunction.apply(A[NETWORKSIZE], targets).apply(true);
            boolean outputLayer = true;
            for (int i = 0; i < NETWORKSIZE; i++) {
                int currentIndex = NETWORKSIZE - 1 - i;//Increase performance by reducing amount of pointers
                Layer currentLayer = network[currentIndex];//Increase performance by reducing amount of pointers
                if (!outputLayer) {
                    dZ_dA = network[currentIndex + 1].weights;
                }
                if (outputLayer) {
                    dA_dZ = activationOutputs.apply(Z[currentIndex], true);
                } else {
                    dA_dZ = activationHiddens.apply(Z[currentIndex], true);
                }
                dZ_dW = A[currentIndex];
                if (!outputLayer) {
                    dC_dA = dotProduct.apply(dC_dZ, transpose(dZ_dA));
                }
                if (outputLayer) {
                    dC_dZ = multiply(dA_dZ, dC_dA);
                } else {
                    dC_dZ = multiply(dC_dA, dA_dZ);
                }
                dC_dW = dotProduct.apply(transpose(dZ_dW), dC_dZ);
                lossFunction.apply(A[NETWORKSIZE], targets).apply(false);//Update the cost to track progress
                //Add optimizer or updater to gradients
                float[][][] updateB = updater.apply(lr, dC_dZ).apply(previousMomentsB[0][currentIndex], previousMomentsB[1][currentIndex]);
                float[][][] updateW = updater.apply(lr, dC_dW).apply(previousMomentsW[0][currentIndex], previousMomentsW[1][currentIndex]);
                bGradients = updateB[0];
                wGradients = updateW[0];
                //Update weights and biases
                currentLayer.biases = subtract(currentLayer.biases, bGradients);
                currentLayer.weights = subtract(currentLayer.weights, wGradients);
                //Save old gradients
                previousMomentsW[0][currentIndex] = updateW[1];
                previousMomentsB[0][currentIndex] = updateB[1];
                previousMomentsW[1][currentIndex] = updateW[2];
                previousMomentsB[1][currentIndex] = updateB[2];
                if (outputLayer) {
                    outputLayer = false;
                }
            }
            sessions++;
            globalCost = cost;
        }

        public void setSeed(long seed) {
            this.seed = seed;
            random.setSeed(seed);
        }

        /**
         *
         * @param optimizer Optimizer for the stochastic gradient
         * descent(momentum).
         */
        public void setOptimizer(String optimizer) {
            if ("".equals(optimizer)) {
                this.updater = (a, b) -> (c, d) -> {
                    return new float[][][]{scale(a, b), null, null};
                };
            } else if ("momentum".equalsIgnoreCase(optimizer)) {
                this.updater = (a, b) -> (c, d) -> momentum(a, b, c);
            } else if ("adam".equalsIgnoreCase(optimizer)) {
                this.updater = (a, b) -> (c, d) -> adam(a, b, c, d);
            } else if ("nadam".equalsIgnoreCase(optimizer)) {
                this.updater = (a, b) -> (c, d) -> nadam(a, b, c, d);
            } else {
                throw new IllegalArgumentException("INVALID OPTIMIZER");
            }
        }

        private float[][][] momentum(float lr, float[][] gradients, float[][] prevUpdate) {
            final float beta = .9f;
            float[][] update = add(scale(beta, prevUpdate), scale(lr, gradients));
            return new float[][][]{update, update, null};
        }

        private float[][][] adam(float lr, float[][] gradients, float[][] moment1, float[][] moment2) {
            final float b1 = .9f;
            final float b2 = .99f;
            final float e = .00000001f;
            float[][] m = add(scale((1 - b1), gradients), scale(b1, moment1));
            float[][] v = add(scale((1 - b2), power(gradients, 2)), scale(b2, moment2));
            float[][] m_ = scale(1 / (1 - b1), m);
            float[][] v_ = scale(1 / (1 - b2), v);
            float[][] update = divide(scale(lr, m_), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][]{update, m, v};
        }

        private float[][][] nadam(float lr, float[][] gradients, float[][] moment1, float[][] moment2) {
            final float b1 = .9f;
            final float b2 = .99f;
            final float e = .00000001f;
            int rows = gradients.length;
            int columns = gradients[0].length;
            float[][] m = add(scale((1 - b1), gradients), scale(b1, moment1));
            float[][] v = add(scale((1 - b2), power(gradients, 2)), scale(b2, moment2));
            float[][] m_ = scale(1 / (1 - b1), m);
            float[][] v_ = scale(1 / (1 - b2), v);
            return new float[][][]{multiply(divide(create(rows, columns, lr), add(sqrt(v_), create(rows, columns, e))), add(scale(b1, m_), scale(scale(1 - b1, gradients), 1 / (1 - b1)))), m, v};
        }

        private float[][] linearActivation(float[][] matrix, boolean derivative) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            if (!derivative) {
                return matrix;
            } else {
                float[][] matrixResult = create(rows, columns, 1);
                return matrixResult;
            }
        }

        private float[][] softmaxActivation(float[][] matrix, boolean derivative) {
            float[][] result = softmax(matrix);
            if (!derivative) {
                return result;
            }
            float[][] ones = create(matrix.length, matrix[0].length, 1);
            float[][] derivatives = multiply(result, subtract(ones, result));
            return derivatives;
        }

        public float[][] softmax(float[][] matrix) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            float sum = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    sum += Math.exp(matrix[i][j]);
                }
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = (float) Math.exp(matrix[i][j]) / sum;
                }
            }
            return matrixResult;
        }

        private float relu(float x, boolean derivative) {
            if (derivative == true) {
                if (x < 0) {
                    return 0;
                } else {
                    return 1;
                }
            } else {
                if (x < 0) {
                    return 0;
                } else {
                    return x;
                }
            }
        }

        private float[][] reluActivation(float[][] matrix, boolean derivative) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = relu(matrix[i][j], derivative);
                }
            }
            return matrixResult;
        }

        private float leakyRelu(float x, boolean derivative) {
            if (derivative == true) {
                if (x < 0) {
                    return .001f;
                } else {
                    return 1;
                }
            } else {
                if (x < 0) {
                    return .001f * x;
                } else {
                    return x;
                }
            }
        }

        private float[][] leakyReluActivation(float[][] matrix, boolean derivative) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = leakyRelu(matrix[i][j], derivative);
                }
            }
            return matrixResult;
        }

        private float sigmoid(float x, boolean derivative) {
            if (derivative == true) {
                return sigmoid(x, false) * (1 - sigmoid(x, false));//sigmoid'(x)
            }
            return 1 / (1 + (float) Math.exp(-x));//sigmoid(x)
        }

        private float[][] sigmoidActivation(float[][] matrix, boolean derivative) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = sigmoid(matrix[i][j], derivative);
                }
            }
            return matrixResult;
        }

        private float tanh(float x, boolean derivative) {
            if (derivative == true) {
                return 1 - ((float) Math.pow(tanh(x, false), 2));//tanh'(x)
            }
            return (2 / (1 + (float) Math.exp(-2 * x))) - 1;//tanh(x)
        }

        private float[][] tanhActivation(float[][] matrix, boolean derivative) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = tanh(matrix[i][j], derivative);
                }
            }
            return matrixResult;
        }

        private float[][] quadraticLoss(float[][] outputs, float[][] targets, boolean derivative) {
            if (derivative) {
                return subtract(outputs, targets);
            }
            float[][] loss = scale(power(subtract(outputs, targets), 2), .5f);
            int rows = loss.length;
            int columns = loss[0].length;
            double total = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    total += loss[i][j];
                }
            }
            cost = total / columns;
            return null;
        }

        private float[][] logLoss(float[][] outputs, float[][] targets, boolean derivative) {
            if (derivative) {
                int rows = outputs.length;
                int columns = outputs[0].length;
                return divide(subtract(outputs, targets), add(multiply(subtract(create(rows, columns, 1), outputs), outputs), create(rows, columns, Float.MIN_VALUE)));//Adding minimum value to prevent dividing by zero
            }
            cost = -sum(multiply(targets, ln(outputs)));
            return null;
        }

        public float[][] normalizeTanhEstimator(float[][] inputs) {
            int inputLength = inputs[0].length;
            float[][] result = new float[1][inputLength];
            float mean = sum(inputs) / inputLength;
            float deviation = (float) (Math.sqrt(sum(power(subtract(inputs, create(1, inputLength, mean)), 2)) / (mean)));
            for (int i = 0; i < inputs[0].length; i++) {
                result[0][i] = (float) (.5 * (tanh((float) (.01 * ((inputs[0][i] - mean) / (deviation))), false) + 1));//Tanh estimator normalization
            }
            return result;
        }

        private void sizeException(float[][] matrix) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            for (int i = 0; i < rows; i++) {
                if (matrix[i].length != columns) {
                    throw new IllegalArgumentException("Inconsistent Matrix Size");
                }
            }
        }

        private void dotDimensionMismatch(float[][] matrixA, float[][] matrixB) {
            if (matrixA[0].length != matrixB.length)//A columns must equal B rows
            {
                throw new IllegalArgumentException("Matrices Dimension Mismatch");
            }
        }

        private void dimensionMismatch(float[][] matrixA, float[][] matrixB) {
            if (matrixA.length != matrixB.length || matrixA[0].length != matrixB[0].length) {
                throw new IllegalArgumentException("Matrices Dimension Mismatch");
            }
        }

        public void print(float[][] matrix, String nameOfMatrix) {
            System.out.println(nameOfMatrix + ": ");
            int rows = matrix.length;
            int columns = matrix[0].length;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    System.out.print("[" + matrix[i][j] + "] ");
                }
                System.out.println("");
            }
        }

        public float[][] doubleToFloat(double[][] matrix) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = (float) (matrix[i][j]);
                }
            }
            return matrixResult;
        }

        public float[][] create(int rows, int columns, float valueToAllElements) {
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = valueToAllElements;
                }
            }
            return matrixResult;
        }

        public float[][] randomize(float[][] matrix, float range, float minimum) {
            sizeException(matrix);
            float[][] matrixResult = matrix;
            int rows = matrixResult.length;
            int columns = matrixResult[0].length;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = random.nextFloat() * range + minimum;
                }
            }
            return matrixResult;
        }

        public float[][] transpose(float[][] matrix) {
            sizeException(matrix);
            float[][] matrixResult = new float[matrix[0].length][matrix.length];
            int rows = matrix.length;
            int columns = matrix[0].length;
            for (int j = 0; j < columns; j++) {
                for (int i = 0; i < rows; i++) {
                    matrixResult[j][i] = matrix[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] scale(float[][] matrix, float factor) {
            sizeException(matrix);
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = factor * matrix[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] scale(float factor, float[][] matrix) {
            sizeException(matrix);
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = factor * matrix[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] add(float[][] matrixA, float[][] matrixB) {
            sizeException(matrixA);
            sizeException(matrixB);
            dimensionMismatch(matrixA, matrixB);
            int rows = matrixA.length;
            int columns = matrixA[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = matrixA[i][j] + matrixB[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] subtract(float[][] matrixA, float[][] matrixB) {
            sizeException(matrixA);
            sizeException(matrixB);
            dimensionMismatch(matrixA, matrixB);
            int rows = matrixA.length;
            int columns = matrixA[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = matrixA[i][j] - matrixB[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] multiply(float[][] matrixA, float[][] matrixB) {
            sizeException(matrixA);
            sizeException(matrixB);
            dimensionMismatch(matrixA, matrixB);
            int rows = matrixA.length;
            int columns = matrixA[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = matrixA[i][j] * matrixB[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] divide(float[][] matrixA, float[][] matrixB) {
            sizeException(matrixA);
            sizeException(matrixB);
            dimensionMismatch(matrixA, matrixB);
            int rows = matrixA.length;
            int columns = matrixA[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = matrixA[i][j] / matrixB[i][j];
                }
            }
            return matrixResult;
        }

        public float[][] dot(float[][] m1, float[][] m2) {
            //make sure lengths of rows are the same for each matrix
            sizeException(m1);
            sizeException(m2);
            dotDimensionMismatch(m1, m2);
            int rows = m1.length;
            int columns = m2[0].length;
            int columns2 = m1[0].length;
            float[][] matrixResult = new float[rows][columns];//result matrix
            //multiply A row elements by B column elements = 1 element in result matrix
            for (int i = 0; i < rows; i++)//A rows or B columns or Result rows
            {
                for (int k = 0; k < columns2; k++)//A rows or B columns or Result columns
                {
                    for (int j = 0; j < columns; j++)//A columns or B rows
                    {
                        matrixResult[i][j] += m1[i][k] * m2[k][j];//Making the j loop the inner most loop improves performance than if k or i is the inner most loop
                    }
                }
            }
            return matrixResult;
        }

        private class MatrixThread extends Thread {

            int num;
            int threadNum;
            int rows;
            int columns;
            int columns2;
            float[][] m1;
            float[][] m2;
            float[][] result;

            MatrixThread(int num, int threadNum, int rows, int columns, int columns2, float[][] m1, float[][] m2, float[][] result) {
                this.num = num;
                this.threadNum = threadNum;
                this.rows = rows;
                this.columns = columns;
                this.columns2 = columns2;
                this.m1 = m1;
                this.m2 = m2;
                this.result = result;
            }

            @Override
            public void run() {
                for (int i = num * rows / threadNum; i < (num + 1) * rows / threadNum; i++) {
                    for (int k = 0; k < columns2; k++) {
                        for (int j = 0; j < columns; j++) {
                            result[i][j] += m1[i][k] * m2[k][j];
                        }
                    }
                }
            }
        }

        public void setThreads(int numberOfThreads) {
            if (numberOfThreads <= 1) {
                dotProduct = (a, b) -> dot(a, b);
            } else {
                threads = numberOfThreads;
                dotProduct = (a, b) -> dotThreads(a, b);
            }
        }

        public float[][] dotThreads(float[][] m1, float[][] m2) {
            sizeException(m1);
            sizeException(m2);
            dotDimensionMismatch(m1, m2);
            MatrixThread[] threadArray = new MatrixThread[threads];
            int rows = m1.length;
            int columns = m2[0].length;
            int columns2 = m1[0].length;
            float[][] result = new float[rows][columns];
            for (int t = 0; t < threads; t++) {
                threadArray[t] = new MatrixThread(t, threads, rows, columns, columns2, m1, m2, result);
                threadArray[t].start();
            }
            for (int i = 0; i < threads; i++) {
                try {
                    threadArray[i].join();
                } catch (Exception e) {

                }
            }
            return result;
        }

        public float[][] power(float[][] matrix, double power) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = (float) Math.pow(matrix[i][j], power);
                }
            }
            return matrixResult;
        }

        public float[][] sqrt(float[][] matrix) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = (float) Math.sqrt(matrix[i][j]);
                }
            }
            return matrixResult;
        }

        public float sum(float[][] matrix) {
            float sum = 0;
            int rows = matrix.length;
            int columns = matrix[0].length;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    sum += matrix[i][j];
                }
            }
            return sum;
        }

        public float[][] ln(float[][] matrix) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] result = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    result[i][j] = (float) Math.log(matrix[i][j]);
                }
            }
            return result;
        }

        public float[][] copy(float[][] matrix) {
            int rows = matrix.length;
            int columns = matrix[0].length;
            float[][] matrixResult = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    matrixResult[i][j] = matrix[i][j];
                }
            }
            return matrixResult;
        }

        public int argmax(float[][] oneRowMatrix) {
            float max = Float.NEGATIVE_INFINITY;
            int index = 0;
            for (int i = 0; i < oneRowMatrix[0].length; i++) {
                if (max < oneRowMatrix[0][i]) {
                    max = oneRowMatrix[0][i];
                    index = i;
                }
            }
            return index;
        }

        public int argmin(float[][] oneRowMatrix) {
            float min = Float.POSITIVE_INFINITY;
            int index = 0;
            for (int i = 0; i < oneRowMatrix[0].length; i++) {
                if (min > oneRowMatrix[0][i]) {
                    min = oneRowMatrix[0][i];
                    index = i;
                }
            }
            return index;
        }

        public float[][] append(float[][] oneRow1, float[][] oneRow2) {
            int length1 = oneRow1[0].length;
            int length2 = oneRow2[0].length;
            float[][] result = new float[1][length1 + length2];
            for (int i = 0; i < length1; i++) {
                result[0][i] = oneRow1[0][i];
            }
            for (int i = 0; i < length2; i++) {
                result[0][i + length1] = oneRow2[0][i];
            }
            return result;
        }
    }

    private static void initGraph(Stage stage) {
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setAnimated(false);
        xAxis.setLabel("Training Sessions");
        yAxis.setAnimated(false);
        yAxis.setLabel(graphMeasuresAccuracy ? "Accuracy" : "Cost");
        if (graphMeasuresAccuracy) {
            yAxis.setAutoRanging(false);
            yAxis.setUpperBound(1);
            yAxis.setTickUnit(.05);
        }
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(yAxis.getLabel() + " over " + xAxis.getLabel());
        ScatterChart<Number, Number> chart = new ScatterChart<>(xAxis, yAxis);
        chart.setAnimated(false);
        chart.getData().add(series);
        Scene scene = new Scene(chart, 600, 300);
        stage.setScene(scene);
        stage.show();
        Thread updateThread = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(50);
                    Platform.runLater(() -> series.getData().add(new XYChart.Data<>(sessions, !graphMeasuresAccuracy ? globalCost : 1 / Math.pow(10, globalCost))));
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        updateThread.setDaemon(true);
        updateThread.start();
    }

    public static void graphJFX(boolean graphMeasuresAccuracy) {//Cost = false, Accuracy = true
        NNest.graphMeasuresAccuracy = graphMeasuresAccuracy;
        Stage stage = new Stage();
        initGraph(stage);
    }

    public static void graph(boolean graphMeasuresAccuracy) {
        NNest.graphMeasuresAccuracy = graphMeasuresAccuracy;
        new Thread(() -> {
            NNest.launch(NNest.class);
        }).start();
    }

    @Override
    public void start(Stage stage) {
        initGraph(stage);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
