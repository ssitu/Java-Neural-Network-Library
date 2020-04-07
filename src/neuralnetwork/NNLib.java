package neuralnetwork;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class NNLib extends Application implements Serializable {

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

    static public enum Optimizer {
        VANILLA, MOMENTUM, RMSPROP, ADAM, ADAMAX, NADAM, AMSGRAD
    }

    private static boolean graphMeasuresAccuracy;
    private static NN nnForGraph;

    public final class NN implements Serializable {

        private class Layer implements Serializable {

            float[][] weights;
            float[][] biases;

            Layer(int nodesIn, int nodesOut, Initializer initializer) {
                weights = create(nodesIn, nodesOut, 0);
                biases = create(1, nodesOut, 0);
                weights = randomize(weights, 2, -1);
                biases = randomize(biases, 2, -1);
                if (null == initializer) {
                    throw new IllegalArgumentException("INVALID INITIALIZATION METHOD");
                } else {
                    switch (initializer) {
                        case VANILLA:
                            weights = vanilla(weights);
                            break;
                        case XAVIER:
                            weights = xavier(weights, nodesIn);
                            break;
                        case HE:
                            weights = he(weights, nodesIn);
                            break;
                        default:
                            throw new IllegalArgumentException("INVALID INITIALIZATION METHOD");
                    }
                }
            }

            private float[][] vanilla(float[][] weights) {
                return weights;
            }

            private float[][] xavier(float[][] weights, int nodesIn) {
                return scale(weights, (float) (1 / Math.sqrt(nodesIn)));
            }

            private float[][] he(float[][] weights, int nodesIn) {
                return scale(weights, (float) Math.sqrt(2 / nodesIn));
            }
        }

        private long sessions = 0;
        private int threads;
        public final String NAME;
        private Layer[] network;
        private Random random = new Random();
        private long seed;
        private final float lr;
        private double cost;
        public final int NETWORKSIZE;//Total layers not including the input layer
        private final Initializer INITIALIZER;
        private ActivationFunction HIDDENACTIVATIONFUNCTION;
        private ActivationFunction OUTPUTACTIVATIONFUNCTION;
        private LossFunction LOSSFUNCTION;
        private Optimizer OPTIMIZER;
        private final int[] LAYERNODES;
        private final float[][][][] previousMomentsW;
        private final float[][][][] previousMomentsB;
        private transient BiFunction<float[][], Boolean, float[][]> activationHiddens;
        private transient BiFunction<float[][], Boolean, float[][]> activationOutputs;
        private transient BiFunction<float[][], float[][], float[][]> lossFunction;
        private transient BiFunction<Float, float[][], BiFunction<float[][], float[][], float[][][]>> optimizer;
        private transient BiFunction<float[][], float[][], float[][]> dotProduct = (a, b) -> dot(a, b);

        NN(String name, long seed, double learningRate, Initializer weightInitializer, ActivationFunction hiddenActivationFunction, ActivationFunction outputActivationFunction, LossFunction lossFunction, Optimizer optimizer, int... layerNodes) {
            NAME = name;
            random.setSeed(seed);
            lr = (float) learningRate;
            INITIALIZER = weightInitializer;
            HIDDENACTIVATIONFUNCTION = hiddenActivationFunction;
            OUTPUTACTIVATIONFUNCTION = outputActivationFunction;
            LOSSFUNCTION = lossFunction;
            OPTIMIZER = optimizer;
            LAYERNODES = layerNodes;
            setActivationFunctionHiddens(HIDDENACTIVATIONFUNCTION);
            setActivationFunctionOutputs(OUTPUTACTIVATIONFUNCTION);
            setLossFunction(LOSSFUNCTION);
            setOptimizer(OPTIMIZER);
            network = new Layer[layerNodes.length - 1];
            for (int i = 1; i < layerNodes.length; i++) {
                Layer layer = new Layer(layerNodes[i - 1], layerNodes[i], INITIALIZER);//Adding each layer
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

        public Layer getNetworkLayer(int layerIndex) {//0 returns the layer after the inputs (first hidden layer). Holds weights between itself and the layer before.
            return network[layerIndex];
        }

        public int getNetworkSize() {
            return network.length;
        }

        public Random getRandom() {
            return random;
        }

        public void setSeed(long seed) {
            this.seed = seed;
            random.setSeed(seed);
        }

        public void setThreads(int numberOfThreads) {
            if (numberOfThreads <= 1) {
                dotProduct = (a, b) -> dot(a, b);
            } else {
                threads = numberOfThreads;
                dotProduct = (a, b) -> dotThreads(a, b);
            }
        }

        @Override
        public NN clone() {
            NN nnCopy = new NN(NAME, seed, lr, INITIALIZER, HIDDENACTIVATIONFUNCTION, OUTPUTACTIVATIONFUNCTION, LOSSFUNCTION, OPTIMIZER, LAYERNODES);
            for (int i = 0; i < getNetworkSize(); i++) {
                nnCopy.getNetworkLayer(i).weights = copy(getNetworkLayer(i).weights);
                nnCopy.getNetworkLayer(i).biases = copy(getNetworkLayer(i).biases);
            }
            return nnCopy;
        }

        public void save() {
            try {
                FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + "/" + NAME + "_neuralnetwork(" + toString() + ")");
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                out.writeObject(this);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public boolean load() {
            try {
                FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + "/" + NAME + "_neuralnetwork(" + toString() + ")");
                ObjectInputStream in = new ObjectInputStream(fileIn);
                NN nn = (NN) in.readObject();
                network = nn.network;
                random = nn.random;
                return true;
            } catch (IOException | ClassNotFoundException e) {
                e.printStackTrace();
                System.out.println("Could not load network settings.");
                return false;
            }
        }

        public float[][] feedforward(float[][] inputs) {
            float[][] outputs = inputs;
            for (int i = 0; i < NETWORKSIZE - 1; i++) {//Feed the inputs through the hidden layers
                Layer currentLayer = network[i];
                outputs = activationHiddens.apply(add(dotProduct.apply(outputs, currentLayer.weights), currentLayer.biases), false);
            }
            Layer lastLayer = network[NETWORKSIZE - 1];
            outputs = activationOutputs.apply(add(dotProduct.apply(outputs, lastLayer.weights), lastLayer.biases), false);//Feed the output from the hidden layers to the output layer with its activation function

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
            float[][][] Z = new float[NETWORKSIZE][][];//"Z" = the unactivated outputs from the weights and biases
            float[][][] A = new float[NETWORKSIZE + 1][][];//"A" = the activated "Z"s
            A[0] = outputs;
            for (int i = 0; i < NETWORKSIZE - 1; i++) {
                Layer currentLayer = network[i];
                outputs = add(dotProduct.apply(outputs, currentLayer.weights), currentLayer.biases);//Computing "Z"
                Z[i] = outputs;
                outputs = activationHiddens.apply(outputs, false);//Computing "A"
                A[i + 1] = outputs;
            }
            outputs = add(dotProduct.apply(outputs, lastLayer.weights), lastLayer.biases);
            Z[NETWORKSIZE - 1] = outputs;
            outputs = activationOutputs.apply(outputs, false);
            A[NETWORKSIZE] = outputs;
            dC_dA = lossFunction.apply(A[NETWORKSIZE], targets);
            boolean outputLayer = true;
            for (int i = 0; i < NETWORKSIZE; i++) {
                int currentIndex = NETWORKSIZE - 1 - i;
                Layer currentLayer = network[currentIndex];
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
                //Add optimizer or updater to gradients
                float[][][] updateB = optimizer.apply(lr, dC_dZ).apply(previousMomentsB[0][currentIndex], previousMomentsB[1][currentIndex]);
                float[][][] updateW = optimizer.apply(lr, dC_dW).apply(previousMomentsW[0][currentIndex], previousMomentsW[1][currentIndex]);
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

        private BiFunction<float[][], Boolean, float[][]> assignActivationFunction(BiFunction<float[][], Boolean, float[][]> activation, ActivationFunction activationFunction, boolean hiddens) {
            String layer;
            if (hiddens) {
                layer = "HIDDEN";
            } else {
                layer = "OUTPUT";
            }
            if (null == activationFunction) {
                throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE " + layer + " LAYER");
            } else {
                switch (activationFunction) {
                    case LINEAR:
                        activation = (a, b) -> activationLinear(a, b);
                        break;
                    case SIGMOID:
                        activation = (a, b) -> activationSigmoid(a, b);
                        break;
                    case TANH:
                        activation = (a, b) -> activationTanh(a, b);
                        break;
                    case RELU:
                        activation = (a, b) -> activationRelu(a, b);
                        break;
                    case LEAKYRELU:
                        activation = (a, b) -> activationLeakyRelu(a, b);
                        break;
                    case SWISH:
                        activation = (a, b) -> activationSwish(a, b);
                        break;
                    case MISH:
                        activation = (a, b) -> activationMish(a, b);
                        break;
                    case CUSTOM:
                        activation = (a, b) -> activationCustom(a, b);
                        break;
                    case SOFTMAX:
                        activation = (a, b) -> activationSoftmax(a, b);
                        break;
                    default:
                        throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE " + layer + " LAYER");
                }
            }
            if (hiddens) {
                HIDDENACTIVATIONFUNCTION = activationFunction;
            } else {
                OUTPUTACTIVATIONFUNCTION = activationFunction;
            }
            return activation;
        }

        public void setActivationFunctionHiddens(ActivationFunction hiddenActivationFunction) {
            activationHiddens = assignActivationFunction(activationHiddens, hiddenActivationFunction, true);
        }

        public void setActivationFunctionOutputs(ActivationFunction outputActivationFunction) {
            activationOutputs = assignActivationFunction(activationOutputs, outputActivationFunction, false);
        }

        public void setLossFunction(LossFunction lossFunction) {
            if (null == lossFunction) {
                throw new IllegalArgumentException("INVALID COST FUNCTION");
            } else {
                switch (lossFunction) {
                    case QUADRATIC:
                        this.lossFunction = (a, b) -> lossQuadratic(a, b);
                        break;
                    case HUBER:
                        this.lossFunction = (a, b) -> lossHuber(a, b);
                        break;
                    case HUBERPSEUDO:
                        this.lossFunction = (a, b) -> lossPseudoHuber(a, b);
                        break;
                    case CUSTOM:
                        this.lossFunction = (a, b) -> lossCustom(a, b);
                        break;
                    case CROSS_ENTROPY:
                        this.lossFunction = (a, b) -> lossLog(a, b);
                        break;
                    default:
                        throw new IllegalArgumentException("INVALID COST FUNCTION");
                }
            }
            LOSSFUNCTION = lossFunction;
        }

        public void setOptimizer(Optimizer updater) {
            if (null == updater) {
                throw new IllegalArgumentException("INVALID OPTIMIZER");
            } else {
                switch (updater) {
                    case VANILLA:
                        optimizer = (a, b) -> (c, d) -> new float[][][]{scale(a, b), null, null};//Scale gradients by the learning rate
                        break;
                    case MOMENTUM:
                        optimizer = (a, b) -> (c, d) -> momentum(a, b, c);
                        break;
                    case RMSPROP:
                        optimizer = (a, b) -> (c, d) -> rmsprop(a, b, c);
                    case ADAM:
                        optimizer = (a, b) -> (c, d) -> adam(a, b, c, d);
                        break;
                    case ADAMAX:
                        optimizer = (a, b) -> (c, d) -> adamax(a, b, c, d);
                    case NADAM:
                        optimizer = (a, b) -> (c, d) -> nadam(a, b, c, d);
                        break;
                    case AMSGRAD:
                        optimizer = (a, b) -> (c, d) -> amsgrad(a, b, c, d);
                        break;
                    default:
                        throw new IllegalArgumentException("INVALID OPTIMIZER");
                }
            }
            OPTIMIZER = updater;
        }

        private float[][][] momentum(float lr, float[][] gradients, float[][] v) {
            final float beta = .9f;
            float[][] update = add(scale(beta, v), scale(lr, gradients));
            return new float[][][]{update, update, null};
        }

        private float[][][] rmsprop(float lr, float[][] gradients, float[][] sPrev) {//https://www.coursera.org/lecture/deep-neural-network/rmsprop-BhJlm
            final float beta = .9f;
            final float e = .00000001f;
            float[][] s = add(scale(beta, sPrev), scale(1 - beta, square(gradients)));
            float[][] update = divide(scale(lr, gradients), add(sqrt(s), create(s.length, s[0].length, e)));
            return new float[][][]{update, sPrev, null};
        }

        private float[][][] adam(float lr, float[][] gradients, float[][] moment1, float[][] moment2) {
            final float beta1 = .9f;
            final float beta2 = .999f;
            final float e = .00000001f;
            float[][] m = add(scale((1 - beta1), gradients), scale(beta1, moment1));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, moment2));
            float[][] m_ = scale(1 / (1 - beta1), m);//debiasing
            float[][] v_ = scale(1 / (1 - beta2), v);//debiasing
            float[][] update = divide(scale(lr, m_), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][]{update, m, v};
        }

        private float[][][] adamax(float lr, float[][] gradients, float[][] moment1, float[][] moment2) {
            final float beta1 = .9f;
            final float beta2 = .999f;
            final float e = .00000001f;
            float[][] m = add(scale((1 - beta1), gradients), scale(beta1, moment1));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, moment2));
            float[][] m_ = scale(1 / (1 - beta1), m);
            float[][] u = max(scale(beta2, moment2), abs(gradients));
            float[][] update = divide(scale(lr, m_), u);
            return new float[][][]{update, m, v};
        }

        private float[][][] nadam(float lr, float[][] gradients, float[][] moment1, float[][] moment2) {
            final float beta1 = .9f;
            final float beta2 = .999f;
            final float e = .00000001f;
            int rows = gradients.length;
            int columns = gradients[0].length;
            float[][] m = add(scale((1 - beta1), gradients), scale(beta1, moment1));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, moment2));
            float[][] m_ = scale(1 / (1 - beta1), m);
            float[][] v_ = scale(1 / (1 - beta2), v);
            float[][] update = multiply(divide(create(rows, columns, lr), add(sqrt(v_), create(rows, columns, e))), add(scale(beta1, m_), scale(scale(1 - beta1, gradients), 1 / (1 - beta1))));
            return new float[][][]{update, m, v};
        }

        private float[][][] amsgrad(float lr, float[][] gradients, float[][] moment1, float[][] moment2) {
            final float beta1 = .9f;
            final float beta2 = .999f;
            final float e = .00000001f;
            float[][] m = add(scale((1 - beta1), gradients), scale(beta1, moment1));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, moment2));
            float[][] v_ = max(moment2, v);
            float[][] update = divide(scale(lr, m), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][]{update, m, v};
        }

        private float[][] activationLinear(float[][] matrix, boolean derivative) {
            if (!derivative) {
                return matrix;
            } else {
                float[][] matrixResult = create(matrix.length, matrix[0].length, 1);
                return matrixResult;
            }
        }

        private float[][] activationSigmoid(float[][] matrix, boolean derivative) {
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

        private float[][] activationTanh(float[][] matrix, boolean derivative) {
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

        private float relu(float x, boolean derivative) {
            if (derivative) {
                if (x < 0) {
                    return 0;
                } else {
                    return 1;
                }
            } else {
                return Math.max(0, x);
            }
        }

        private float[][] activationRelu(float[][] matrix, boolean derivative) {
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
                return Math.max(.001f * x, x);
            }
        }

        private float[][] activationLeakyRelu(float[][] matrix, boolean derivative) {
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

        private float swish(float x, boolean derivative) {
            if (!derivative) {
                return x * sigmoid(x, false);
            } else {
                return x * sigmoid(x, true) + sigmoid(x, false);
            }
        }

        private float[][] activationSwish(float[][] m, boolean derivative) {
            int rows = m.length;
            int columns = m[0].length;
            float[][] result = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    result[i][j] = swish(m[i][j], derivative);
                }
            }
            return result;
        }

        private float mish(float x, boolean derivative) {
            if (!derivative) {
                return (float) (x * tanh((float) Math.log(1 + Math.exp(x)), false));
            } else {
                double x_ = (double) x;
                return (float) ((Math.exp(x_) * ((4 * (x_ + 1)) + (4 * Math.exp(2 * x_)) + (Math.exp(3 * x_)) + (Math.exp(x_) * (4 * x_ + 6)))) / ((2 * Math.exp(2 * x_)) + (Math.exp(2 * x_)) + 2));
            }
        }

        private float[][] activationMish(float[][] m, boolean derivative) {
            int rows = m.length;
            int columns = m[0].length;
            float[][] result = new float[rows][columns];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    result[i][j] = mish(m[i][j], derivative);
                }
            }
            return result;
        }

        private float[][] activationCustom(float[][] x, boolean derivative) {
            int rows = x.length;
            int columns = x[0].length;
            if (!derivative) {
                return max(activationTanh(x, false), x);
            } else {
                float[][] result = new float[rows][columns];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < columns; j++) {
                        float val = x[i][j];
                        if (tanh(val, false) > val) {
                            result[i][j] = tanh(val, true);
                        } else {
                            result[i][j] = 1;
                        }
                    }
                }
                return result;
            }
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

        private float[][] activationSoftmax(float[][] matrix, boolean derivative) {
            float[][] result = softmax(matrix);
            if (!derivative) {
                return result;
            }
            float[][] ones = create(matrix.length, matrix[0].length, 1);
            float[][] derivatives = multiply(result, subtract(ones, result));
            return derivatives;
        }

        private float[][] lossQuadratic(float[][] outputs, float[][] targets) {
            final float factor = LOSSFUNCTION.steepness;
            float[][] loss = scale(factor, square(subtract(outputs, targets)));//m(f(x) - y)^2 where f(x) is the output of the network and y is the target output
            int columns = loss[0].length;
            double total = 0;
            for (int j = 0; j < columns; j++) {
                total += loss[0][j];
            }
            cost = total / columns;
            return scale(2 * factor, subtract(outputs, targets));
        }

        private float[][] lossHuber(float[][] outputs, float[][] targets) {//Notation at https://en.wikipedia.org/wiki/Huber_loss
            final float delta = LOSSFUNCTION.steepness;
            final float deltaHalf = delta / 2;
            int columns = outputs[0].length;
            float[][] a = subtract(outputs, targets);
            float sum = 0;
            for (int j = 0; j < columns; j++) {
                float val = a[0][j];
                if (Math.abs(val) < delta) {
                    sum += val * val / 2;
                } else {
                    sum += delta * (Math.abs(a[0][j]) - deltaHalf);
                }
            }
            cost = sum / columns;
            float[][] deriv = new float[1][columns];
            for (int j = 0; j < columns; j++) {
                float val = a[0][j];
                if (Math.abs(val) < delta) {
                    deriv[0][j] = a[0][j];
                } else {
                    deriv[0][j] = delta * (a[0][j] / Math.abs(a[0][j])) - delta;
                }
            }
            return deriv;
        }

        private float[][] lossPseudoHuber(float[][] outputs, float[][] targets) {//https://en.wikipedia.org/wiki/Huber_loss
            final float delta = LOSSFUNCTION.steepness;
            int columns = outputs[0].length;
            final float deltaSquared = delta * delta;
            final float[][] ones = create(1, columns, 1);
            final float[][] a = subtract(outputs, targets);
            final float[][] root = sqrt(add(ones, scale(square(a), 1 / deltaSquared)));
            cost = sum(scale(deltaSquared, subtract(root, ones)));
            return divide(a, root);
        }

        private float[][] lossCustom(float[][] outputs, float[][] targets) {
            int columns = outputs[0].length;
            final float[][] a = subtract(outputs, targets);
            final float[][] aSquared = square(a);
            final float[][] constant = create(1, columns, .25f);
            cost = sum(divide(a, add(constant, aSquared))) / columns;//Loss Function
            return divide(subtract(constant, aSquared), square(add(aSquared, constant)));//Derivative of the function with respect to a
        }

        private float[][] lossLog(float[][] outputs, float[][] targets) {
            cost = -sum(multiply(targets, ln(outputs)));//Update cost
            int rows = outputs.length;
            int columns = outputs[0].length;
            return divide(subtract(outputs, targets), add(multiply(subtract(create(rows, columns, 1), outputs), outputs), create(rows, columns, Float.MIN_VALUE)));//Adding minimum value to prevent dividing by zero
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

        public float[][] dotThreads(float[][] m1, float[][] m2) {
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
                } catch (InterruptedException e) {

                }
            }
            return result;
        }
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

    static public void print(float[][] matrix, String nameOfMatrix) {
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

    static public float[][] doubleToFloat(double[][] matrix) {
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

    static public float[][] create(int rows, int columns, float valueToAllElements) {
        float[][] matrixResult = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrixResult[i][j] = valueToAllElements;
            }
        }
        return matrixResult;
    }

    static public float[][] randomize(float[][] matrix, float range, float minimum) {
        float[][] matrixResult = matrix;
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrixResult[i][j] = (float) Math.random() * range + minimum;
            }
        }
        return matrixResult;
    }

    static public float[][] randomize(float[][] matrix, float range, float minimum, Random random) {
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

    static public float[][] transpose(float[][] matrix) {
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

    static public float[][] scale(float[][] matrix, float factor) {
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

    static public float[][] scale(float factor, float[][] matrix) {
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

    static public float[][] add(float[][] matrixA, float[][] matrixB) {
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

    static public float[][] subtract(float[][] matrixA, float[][] matrixB) {
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

    static public float[][] multiply(float[][] matrixA, float[][] matrixB) {
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

    static public float[][] divide(float[][] matrixA, float[][] matrixB) {
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

    static public float[][] dot(float[][] m1, float[][] m2) {
        int rows1 = m1.length;
        int columns1 = m1[0].length;
        int columns2 = m2[0].length;
        float[][] result = new float[rows1][columns2];
        if (columns1 % 4 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < columns1; k += 4) {
                    for (int j = 0; j < columns2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j]
                                + m1[i][k + 2] * m2[k + 2][j]
                                + m1[i][k + 3] * m2[k + 3][j];
                    }
                }
            }
        } else if (columns1 % 3 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < columns1; k += 3) {
                    for (int j = 0; j < columns2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j]
                                + m1[i][k + 2] * m2[k + 2][j];
                    }
                }
            }
        } else if (columns1 % 2 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < columns1; k += 2) {
                    for (int j = 0; j < columns2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j];
                    }
                }
            }
        } else {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < columns1; k++) {
                    for (int j = 0; j < columns2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
        }
        return result;
    }

    static public float[][] power(float[][] matrix, double power) {
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

    static public float[][] square(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] matrixResult = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val = matrix[i][j];
                matrixResult[i][j] = val * val;
            }
        }
        return matrixResult;
    }

    static public float[][] sqrt(float[][] matrix) {
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

    static public float sum(float[][] matrix) {
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

    static public float[][] ln(float[][] matrix) {
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

    static public float[][] copy(float[][] matrix) {
        return Arrays.stream(matrix).map(el -> el.clone()).toArray(a -> matrix.clone());
    }

    static public float[][] oneHot(float[][] m) {
        int rows = m.length;
        int columns = m[0].length;
        float[][] result = create(rows, columns, 0);
        float max = Float.NEGATIVE_INFINITY;
        int x = 0;
        int y = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (max < m[i][j]) {
                    max = m[i][j];
                    x = i;
                    y = j;
                }
            }
        }
        result[x][y] = 1;
        return result;
    }

    static public float[][] abs(float[][] m) {
        int rows = m.length;
        int columns = m[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = Math.abs(m[i][j]);
            }
        }
        return result;
    }

    static public float[][] max(float[][] m1, float[][] m2) {//Size must match
        int rows = m1.length;
        int columns = m1[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val1 = m1[i][j];
                float val2 = m2[i][j];
                if (val1 > val2) {
                    result[i][j] = val1;
                } else {
                    result[i][j] = val2;
                }
            }
        }
        return result;
    }

    static public int argmax(float[][] oneRowMatrix) {
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

    static public int argmin(float[][] oneRowMatrix) {
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

    static public float[][] append(float[][] oneRow1, float[][] oneRow2) {
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

    static public float[][] normalTanh(float[][] inputs) {
        int elements = inputs[0].length;
        float[][] result = new float[1][elements];
        float mean = sum(inputs) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(inputs, create(1, elements, mean)))) / (mean)));
        for (int i = 0; i < inputs[0].length; i++) {
            result[0][i] = (float) (.5 * (tanh((float) (.01 * ((inputs[0][i] - mean) / (deviation))), false) + 1));//Tanh estimator normalization
        }
        return result;
    }

    static public float[][] normalZScore(float[][] inputs) {
        int elements = inputs[0].length;
        float mean = sum(inputs) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(inputs, create(1, elements, mean)))) / (mean)));
        return divide(subtract(inputs, create(1, elements, mean)), create(1, elements, deviation));
    }

    static public float sigmoid(float x, boolean derivative) {
        if (derivative) {
            return sigmoid(x, false) * (1 - sigmoid(x, false));//sigmoid'(x)
        }
        return 1 / (1 + (float) Math.exp(-x));//sigmoid(x)
    }

    static public float tanh(float x, boolean derivative) {
        if (derivative == true) {
            float val = tanh(x, false);
            return 1 - val * val;//tanh'(x)
        }
        return (2 / (1 + (float) Math.exp(-2 * x))) - 1;//tanh(x)
    }

    private static void initGraph(Stage stage, NN nn) {
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
        stage.setTitle(nn.NAME);
        Thread updateThread = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(50);
                    Platform.runLater(() -> series.getData().add(new XYChart.Data<>(nn.sessions, !graphMeasuresAccuracy ? nn.cost : 1 / Math.pow(10, nn.cost))));
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        updateThread.setDaemon(true);
        updateThread.start();
    }

    public static void graphJFX(boolean graphMeasuresAccuracy, NN nnForGraph) {//Cost = false, Accuracy = true
        NNLib.graphMeasuresAccuracy = graphMeasuresAccuracy;
        Stage stage = new Stage();
        initGraph(stage, nnForGraph);
    }

    public static void graph(boolean graphMeasuresAccuracy, NN nnForGraph) {
        NNLib.graphMeasuresAccuracy = graphMeasuresAccuracy;
        NNLib.nnForGraph = nnForGraph;
        new Thread(() -> {
            NNLib.launch(NNLib.class);
        }).start();
    }

    @Override
    public void start(Stage stage) {
        initGraph(stage, nnForGraph);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
