package neuralnetwork;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.FlowPane;
import javafx.scene.text.Text;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.util.Duration;

public class NNLib extends Application implements Serializable {

    public interface Function<T, R> extends java.util.function.Function<T, R>, Serializable {
    }

    public interface BiFunction<T, S, R> extends java.util.function.BiFunction<T, S, R>, Serializable {
    }

    public interface TriFunction<T, S, U, R> extends Serializable {

        R apply(T t, S s, U u);
    }
    private static int threads;
    private static BiFunction<float[][], float[][], float[][]> dotProduct = (a, b) -> dot(a, b);

    public final class NN implements Serializable {

        public final String label;
        public final int length;//Counts only the hidden layers and the output layer, so the input layer doesn't count.
        private Layer[] network;
        private float lr;
        private Random random = new Random();
        private long seed;
        private double loss;
        private long sessions = 0;
        private BiFunction<float[][], float[][], Object[]> lossFunction;
        private TriFunction<Float, float[][], float[][][], float[][][][]> optimizer;

        NN(String networkName, long seed, double learningRate, BiFunction<float[][], float[][], Object[]> lossFunction, TriFunction<Float, float[][], float[][][], float[][][][]> optimizer, Layer... layers) {
            label = networkName;
            this.seed = seed;
            lr = (float) learningRate;
            this.lossFunction = lossFunction;
            this.optimizer = optimizer;
            network = layers;
            length = network.length;
            random.setSeed(seed);
            for (int i = 0; i < length; i++) {
                network[i].initialize(random);
            }
        }

        public float[][] feedforward(float[][] inputs) {
            float[][] out = inputs;
            for (int i = 0; i < length; i++) {
                out = network[i].forward(out);
            }
            return out;
        }

        public void backpropagation(float[][] inputs, float[][] targets) {
            float[][] out = inputs;
            for (int i = 0; i < length; i++) {
                out = network[i].forwardBack(out);
            }
            Object[] lossArr = lossFunction.apply(out, targets);
            loss = (double) lossArr[0];
            float[][] dC_dA = (float[][]) lossArr[1];
            float[][] dC_dZ = network[length - 1].back(dC_dA, null, lr, optimizer, true);
            for (int i = length - 2; i >= 0; i--) {
                network[i].back(dC_dZ, ((Layer.Dense) network[i + 1]).weights, lr, optimizer, false);
            }
            sessions++;
        }

        @Override
        public String toString() {
            String networkLayers = "";
            networkLayers += network[0].nodesIn + ",";
            for (int i = 0; i < length - 1; i++) {
                networkLayers += network[i].nodesOut + ",";
            }
            networkLayers += network[length - 1].nodesOut;
            return networkLayers;
        }

        /**
         * @param layerIndex 0 refers to the first hidden layer following the
         * inputs. Passing in one less than the length of the network would
         * return the output layer.
         * @return The corresponding layer.
         */
        public Layer getLayer(int layerIndex) {
            return network[layerIndex];
        }

        public Random getRandom() {
            return random;
        }

        public void setSeed(long seed) {
            this.seed = seed;
            random.setSeed(seed);
        }

        @Override
        public NN clone() {
            Layer[] layers = new Layer[length];
            for (int i = 0; i < length; i++) {
                layers[i] = network[i].clone();
            }
            NN clone = new NN(label, seed, lr, lossFunction, optimizer, layers);
            return clone;
        }

        public void save() {
            try {
                FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + "/" + label + "_neuralnetwork(" + toString() + ")");
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                Object[] arr = {network, random};
                out.writeObject(arr);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public boolean load() {
            try {
                FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + "/" + label + "_neuralnetwork(" + toString() + ")");
                ObjectInputStream in = new ObjectInputStream(fileIn);
                Object[] arr = (Object[]) in.readObject();
                network = (Layer[]) arr[0];
                random = (Random) arr[1];
                return true;
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("Could not load network settings for \"" + label + "\".");
                return false;
            }
        }

        public void randomize(float range) {
            for (int i = 0; i < length; i++) {
                network[i].randomize(range);
            }
        }

        public void mutate(float range, float mutateRate) {
            for (int i = 0; i < length; i++) {
                network[i].mutate(range, mutateRate);
            }
        }

        public void setLossFunction(BiFunction<float[][], float[][], Object[]> lossFunction) {
            this.lossFunction = lossFunction;
        }

        /**
         * @param optimizer The first two parameters of the TriFunction are
         * strictly for the learning rate and the gradients respectively. The
         * third parameter is an array of matrices to store info such as
         * previous updates, gradients, etc. The TriFunction returns an array
         * where its first element is a array with one element to hold the
         * gradient update matrix and its second element is the storage array
         * that is passed into the next call of the optimizer
         */
        public void setOptimizer(TriFunction<Float, float[][], float[][][], float[][][][]> optimizer) {
            this.optimizer = optimizer;
        }
    }

    public abstract static class Layer implements Serializable {

        int nodesIn;
        int nodesOut;

        abstract void initialize(Random random);

        abstract float[][] forward(float[][] in);

        abstract float[][] forwardBack(float[][] in);

        abstract float[][] back(float[][] dG, float[][] dZ_dA, float lr, TriFunction<Float, float[][], float[][][], float[][][][]> optimizer, boolean outputLayer);

        abstract void randomize(float range);

        abstract void mutate(float range, float mutateRate);

        protected abstract Layer clone();

        @Override
        public abstract String toString();

        public static class Dense extends Layer implements Serializable {

            float[][] weights;
            float[][] biases;
            float[][][] updateStorageW;
            float[][][] updateStorageB;
            BiFunction<float[][], Boolean, float[][]> activation;
            BiFunction<float[][], Integer, float[][]> initializer;
            float[][] Z;
            float[][] prevA;

            Dense(int nodesIn, int nodesOut, BiFunction<float[][], Boolean, float[][]> activation, BiFunction<float[][], Integer, float[][]> initializer) {
                this.nodesIn = nodesIn;
                this.nodesOut = nodesOut;
                this.activation = activation;
                this.initializer = initializer;
            }

            @Override
            void initialize(Random random) {
                weights = create(nodesIn, nodesOut, 0);
                biases = create(1, nodesOut, 0);
                weights = NNLib.randomize(weights, 2, -1, random);//values on interval [-1,1]
                biases = NNLib.randomize(biases, 2, -1, random);//values on interval [-1,1]
                weights = initializer.apply(weights, nodesIn);
                updateStorageW = new float[][][]{create(nodesIn, nodesOut, 0)};
                updateStorageB = new float[][][]{create(1, nodesOut, 0)};
            }

            @Override
            float[][] forward(float[][] in) {
                return activation.apply(add(dotProduct.apply(in, weights), biases), false);
            }

            @Override
            float[][] forwardBack(float[][] in) {
                prevA = in;
                Z = add(dotProduct.apply(in, weights), biases);
                return activation.apply(Z, false);
            }

            @Override
            float[][] back(float[][] dG, float[][] dZ_dA, float lr, TriFunction<Float, float[][], float[][][], float[][][][]> optimizer, boolean outputLayer) {//dG = The running gradient from the previous backpropagated layer or loss function
                float[][] dA_dZ = activation.apply(Z, true);
                float[][] dC_dZ;
                if (!outputLayer) {
                    float[][] dC_dA = dotProduct.apply(dG, transpose(dZ_dA));
                    if (dA_dZ.length == 1) {
                        dC_dZ = multiply(dC_dA, dA_dZ);
                    } else {
                        dC_dZ = dotProduct.apply(dC_dA, dA_dZ);//For jacobian matrices
                    }
                } else {
                    if (dA_dZ.length == 1) {
                        dC_dZ = multiply(dG, dA_dZ);
                    } else {
                        dC_dZ = dotProduct.apply(dG, dA_dZ);//For jacobian matrices
                    }
                }
                float[][] dC_dW = dotProduct.apply(transpose(prevA), dC_dZ);//prevA = dZ_dW;
                float[][][][] updateW = {};
                while (true) {
                    try {
                        updateW = optimizer.apply(lr, dC_dW, updateStorageW);
                        break;
                    } catch (Exception e) {
                        updateStorageW = append(updateStorageW, new float[][][]{create(nodesIn, nodesOut, 0)});
                        updateStorageB = append(updateStorageB, new float[][][]{create(1, nodesOut, 0)});
                    }
                }
                float[][][][] updateB = optimizer.apply(lr, dC_dZ, updateStorageB);
                float[][] gradientsW = updateW[0][0];
                float[][] gradientsB = updateB[0][0];
                updateStorageW = updateW[1];
                updateStorageB = updateB[1];
                weights = subtract(weights, gradientsW);
                biases = subtract(biases, gradientsB);
                return dC_dZ;
            }

            @Override
            public Dense clone() {
                Dense copy = new Dense(weights.length, weights[0].length, activation, Initializer.VANILLA);
                copy.weights = copy(weights);
                copy.biases = copy(biases);
                copy.updateStorageW = copy3d(updateStorageW);
                copy.updateStorageB = copy3d(updateStorageB);
                return copy;
            }

            @Override
            void randomize(float range) {
                weights = NNLib.randomize(weights, range, -range / 2);//values on interval [-1,1]
                biases = NNLib.randomize(biases, range, -range / 2);//values on interval [-1,1]
            }

            @Override
            void mutate(float range, float mutateRate) {
                for (int i = 0; i < nodesIn; i++) {
                    for (int j = 0; j < nodesOut; j++) {
                        if (Math.random() < mutateRate) {
                            weights[i][j] += (float) (Math.random() * range - range / 2);
                        }
                    }
                }
                for (int i = 0; i < nodesOut; i++) {
                    if (Math.random() < mutateRate) {
                        biases[0][i] += (float) (Math.random() * range - range / 2);
                    }
                }
            }

            @Override
            public String toString() {
                return "Weights:\n" + matrixToString(weights) + "\nBiases:\n" + matrixToString(biases);
            }
        }
    }

    public static class Initializer {

        static final BiFunction<float[][], Integer, float[][]> VANILLA = (a, b) -> a;//No change
        static final BiFunction<float[][], Integer, float[][]> XAVIER = (a, b) -> scale(a, (float) (1 / Math.sqrt(b)));
        static final BiFunction<float[][], Integer, float[][]> HE = (a, b) -> scale(a, (float) Math.sqrt(2 / b));
    }

    public static class ActivationFunction {

        static final BiFunction<float[][], Boolean, float[][]> LINEAR = (matrix, derivative) -> {
            if (!derivative) {
                return matrix;
            }
            float[][] result = create(matrix.length, matrix[0].length, 1);
            return result;
        };
        static final BiFunction<float[][], Boolean, float[][]> SIGMOID = (matrix, derivative) -> function(matrix, val -> sigmoid(val, derivative));
        static final BiFunction<float[][], Boolean, float[][]> TANH = (matrix, derivative) -> function(matrix, val -> tanh(val, derivative));
        static final BiFunction<float[][], Boolean, float[][]> RELU = (matrix, derivative) -> function(matrix, val -> relu(val, derivative));
        static final BiFunction<float[][], Boolean, float[][]> LEAKYRELU = (matrix, derivative) -> function(matrix, val -> leakyrelu(val, derivative));
        static final BiFunction<float[][], Boolean, float[][]> SWISH = (matrix, derivative) -> function(matrix, val -> swish(val, derivative));
        static final BiFunction<float[][], Boolean, float[][]> MISH = (matrix, derivative) -> function(matrix, val -> mish(val, derivative));
        static final BiFunction<float[][], Boolean, float[][]> SOFTMAX = (matrix, derivative) -> {
            if (!derivative) {
                return softmax(matrix);
            }
            float[][] softmax = softmax(matrix);
            int columns = matrix[0].length;
            float[][] jacobian = new float[columns][columns];
            for (int i = 0; i < columns; i++) {
                for (int j = 0; j < columns; j++) {
                    if (i == j) {
                        jacobian[i][j] = softmax[0][i] * (1 - softmax[0][j]);
                    } else {
                        jacobian[i][j] = softmax[0][i] * -softmax[0][j];
                    }
                }
            }
            return jacobian;//Not sure if it should be transposed for my transposed style of a neural network, but the matrix is the same transposed in the xor classification example
        };
    }

    public static class LossFunction {//Not sure if the sums should be divided by the number of outputs of the network

        /**
         * @param steepness default value is 0.5
         */
        static final BiFunction<float[][], float[][], Object[]> QUADRATIC(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                double loss = sum(scale(steepness, square(subtract(outputs, targets))));//m(f(x) - y)^2 where f(x) is the output of the network and y is the target output
                return new Object[]{loss, scale(2 * steepness, subtract(outputs, targets))};//Derivative of the loss function for each sample, 2m(f(x) - y)
            };
        }

        /**
         * @param steepness default value is 1
         */
        static final BiFunction<float[][], float[][], Object[]> HUBER(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            final float deltaHalf = steepness / 2;
            return (outputs, targets) -> {
                int columns = outputs[0].length;
                float[][] a = subtract(outputs, targets);
                float sum = 0;
                for (int j = 0; j < columns; j++) {
                    float val = a[0][j];
                    if (Math.abs(val) < steepness) {
                        sum += val * val / 2;
                    } else {
                        sum += steepness * (Math.abs(a[0][j]) - deltaHalf);
                    }
                }
                double loss = sum / columns;
                float[][] deriv = new float[1][columns];
                for (int j = 0; j < columns; j++) {
                    float val = a[0][j];
                    if (Math.abs(val) < steepness) {
                        deriv[0][j] = a[0][j];
                    } else {
                        deriv[0][j] = steepness * (a[0][j] / Math.abs(a[0][j])) - steepness;
                    }
                }
                return new Object[]{loss, deriv};
            };
        }

        /**
         * @param steepness default value is 1
         */
        static final BiFunction<float[][], float[][], Object[]> HUBERPSEUDO(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                int columns = outputs[0].length;
                final float deltaSquared = steepness * steepness;
                final float[][] ones = create(1, columns, 1);
                final float[][] a = subtract(outputs, targets);
                final float[][] root = sqrt(add(ones, scale(square(a), 1 / deltaSquared)));
                double loss = sum(scale(deltaSquared, subtract(root, ones)));
                return new Object[]{loss, divide(a, root)};
            };
        }

        /**
         * @param steepness default value is 1
         */
        static final BiFunction<float[][], float[][], Object[]> CROSSENTROPY(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                double loss = steepness * -sum(multiply(targets, ln(outputs)));
                return new Object[]{loss, scale(-steepness, divide(targets, outputs))};
            };
        }
    }

    public static class Optimizer {

        //Not sure if these should be final
        public static final float beta = .9f;
        public static final float beta2 = .999f;
        public static final float e = .00000001f;
        static final TriFunction<Float, float[][], float[][][], float[][][][]> VANILLA = (lr, gradients, storage) -> {
            return new float[][][][]{{scale(lr, gradients)}, null};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> MOMENTUM = (lr, gradients, storage) -> {
            float[][] update = add(scale(beta, storage[0]), scale(lr, gradients));
            return new float[][][][]{{update}, {update}};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> NESTEROV = (lr, gradients, storage) -> {//Dozat's modification because calculating two gradients would take a lot of recoding
            float[][] m = add(scale(beta, storage[0]), scale(lr, gradients));
            float[][] update = add(scale(beta, m), scale(lr, gradients));
            return new float[][][][]{{update}, {m}};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> RMSPROP = (lr, gradients, storage) -> {
            float[][] s = add(scale(beta, storage[0]), scale(1 - beta, square(gradients)));
            float[][] update = divide(scale(lr, gradients), add(sqrt(s), create(s.length, s[0].length, e)));
            return new float[][][][]{{update}, {s}};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> ADAM = (lr, gradients, storage) -> {
            float[][] m = add(scale((1 - beta), gradients), scale(beta, storage[0]));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, storage[1]));
            float[][] m_ = scale(1 / (1 - beta), m);//debiasing
            float[][] v_ = scale(1 / (1 - beta2), v);//debiasing
            float[][] update = divide(scale(lr, m_), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][][]{{update}, {m, v}};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> ADAMAX = (lr, gradients, storage) -> {
            float[][] m = add(scale(beta, storage[0]), scale(1 - beta, gradients));
            float[][] v = add(scale(beta2, storage[1]), scale(1 - beta2, square(gradients)));
            float[][] m_ = scale(m, 1 / (1 - beta));
            float[][] u = max(scale(beta2, storage[1]), abs(gradients));
            float[][] update = divide(scale(lr, m_), add(u, create(u.length, u[0].length, e)));
            return new float[][][][]{{update}, {m, v}};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> NADAM = (lr, gradients, storage) -> {
            int rows = gradients.length;
            int columns = gradients[0].length;
            float[][] m = add(scale((1 - beta), gradients), scale(beta, storage[0]));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, storage[1]));
            float[][] m_ = scale(1 / (1 - beta), m);
            float[][] v_ = scale(1 / (1 - beta2), v);
            float[][] update = multiply(divide(create(rows, columns, lr), add(sqrt(v_), create(rows, columns, e))), add(scale(beta, m_), scale(scale(1 - beta, gradients), 1 / (1 - beta))));
            return new float[][][][]{{update}, {m, v}};
        };
        static final TriFunction<Float, float[][], float[][][], float[][][][]> AMSGRAD = (lr, gradients, storage) -> {
            float[][] m = add(scale((1 - beta), gradients), scale(beta, storage[0]));
            float[][] v = add(scale((1 - beta2), square(gradients)), scale(beta2, storage[1]));
            float[][] v_ = max(storage[1], v);
            float[][] update = divide(scale(lr, m), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][][]{{update}, {m, v}};
        };
    }

    public static float[][] normalTanh(float[][] inputs) {
        int elements = inputs[0].length;
        float[][] result = new float[1][elements];
        float mean = sum(inputs) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(inputs, create(1, elements, mean)))) / mean));
        for (int i = 0; i < inputs[0].length; i++) {
            result[0][i] = (.5f * (tanh((.01f * ((inputs[0][i] - mean) / (deviation))), false) + 1));//Tanh estimator normalization
        }
        return result;
    }

    public static float[][] normalZScore(float[][] inputs) {
        int elements = inputs[0].length;
        float mean = sum(inputs) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(inputs, create(1, elements, mean)))) / (mean)));
        return divide(subtract(inputs, create(1, elements, mean)), create(1, elements, deviation));
    }

    public static float sigmoid(float x, boolean derivative) {
        if (!derivative) {
            return 1 / (1 + (float) Math.exp(-x));//sigmoid(x)
        }
        return sigmoid(x, false) * (1 - sigmoid(x, false));//sigmoid'(x)
    }

    public static float tanh(float x, boolean derivative) {
        if (!derivative) {
            return (2 / (1 + (float) Math.exp(-2 * x))) - 1;//tanh(x)
        }
        float val = tanh(x, false);
        return 1 - val * val;//tanh'(x)
    }

    public static float relu(float x, boolean derivative) {
        if (!derivative) {
            return Math.max(0, x);
        }
        if (x < 0) {
            return 0;
        }
        return 1;
    }

    public static float leakyrelu(float x, boolean derivative) {
        if (derivative) {
            return Math.max(.001f * x, x);
        }
        if (x < 0) {
            return .001f;
        }
        return 1;
    }

    public static float swish(float x, boolean derivative) {
        if (!derivative) {
            return x * sigmoid(x, false);
        }
        return x * sigmoid(x, true) + sigmoid(x, false);
    }

    public static float mish(float x, boolean derivative) {
        if (!derivative) {
            return (x * tanh((float) Math.log(1 + Math.exp(x)), false));
        }
        double x_ = (double) x;
        return (float) ((Math.exp(x_) * ((4 * (x_ + 1)) + (4 * Math.exp(2 * x_)) + (Math.exp(3 * x_)) + (Math.exp(x_) * (4 * x_ + 6)))) / ((2 * Math.exp(2 * x_)) + (Math.exp(2 * x_)) + 2));
    }

    public static float[][] softmax(float[][] matrix) {
        float[][] e = exp(matrix);
        return scale(1 / sum(e), e);
    }

    public static void setThreads(int numberOfThreads) {
        if (numberOfThreads <= 1) {
            dotProduct = (a, b) -> dot(a, b);
        } else {
            threads = numberOfThreads;
            dotProduct = (a, b) -> dotThreads(a, b);
        }
    }

    private static class MatrixThread extends Thread {

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

    public static float[][] dotThreads(float[][] m1, float[][] m2) {
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

    public static String matrixToString(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        String string = "";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                string += "[" + matrix[i][j] + "] ";
            }
            string += "\n";
        }
        return string;
    }

    public static void print(float[][] matrix) {
        System.out.print(matrixToString(matrix));
    }

    public static float[][] doubleToFloat(double[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = (float) (matrix[i][j]);
            }
        }
        return result;
    }

    public static float[][] create(int rows, int columns, float valueToAllElements) {
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = valueToAllElements;
            }
        }
        return result;
    }

    public static float[][] randomize(float[][] matrix, float range, float minimum) {
        return function(matrix, val -> (float) Math.random() * range + minimum);
    }

    public static float[][] randomize(float[][] matrix, float range, float minimum, Random random) {
        return function(matrix, val -> random.nextFloat() * range + minimum);
    }

    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = new float[columns][rows];
        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public static float[][] scale(float[][] matrix, float factor) {
        return function(matrix, val -> factor * val);
    }

    public static float[][] scale(float factor, float[][] matrix) {
        return function(matrix, val -> factor * val);
    }

    public static float[][] add(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 + val2);
    }

    public static float[][] subtract(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 - val2);
    }

    public static float[][] multiply(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 * val2);
    }

    public static float[][] divide(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 / val2);
    }

    public static float[][] dot(float[][] m1, float[][] m2) {
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

    public static float[][] power(float[][] matrix, double power) {
        return function(matrix, val -> (float) Math.pow(val, power));
    }

    public static float[][] square(float[][] matrix) {
        return function(matrix, val -> val * val);
    }

    public static float[][] sqrt(float[][] matrix) {
        return function(matrix, val -> (float) Math.sqrt(val));
    }

    public static float sum(float[][] matrix) {
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

    public static float[][] abs(float[][] matrix) {
        return function(matrix, val -> Math.abs(val));
    }

    public static float[][] exp(float[][] matrix) {
        return function(matrix, val -> (float) Math.exp(val));
    }

    public static float[][] ln(float[][] matrix) {
        return function(matrix, val -> (float) Math.log(val));
    }

    public static float[][] copy(float[][] matrix) {
        return Arrays.stream(matrix).map(el -> el.clone()).toArray(a -> matrix.clone());
    }

    public static float[][][] copy3d(float[][][] m3d) {
        return Arrays.stream(m3d).map(m2d -> copy(m2d)).toArray(a -> m3d.clone());
    }

    public static float[][] oneHot(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = create(rows, columns, 0);
        float max = Float.NEGATIVE_INFINITY;
        int x = 0;
        int y = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (max < matrix[i][j]) {
                    max = matrix[i][j];
                    x = i;
                    y = j;
                }
            }
        }
        result[x][y] = 1;
        return result;
    }

    public static float[][] max(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> Math.max(val1, val2));
    }

    public static float max(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float max = Float.MIN_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val = matrix[i][j];
                if (val < max) {
                    max = val;
                }
            }
        }
        return max;
    }

    public static int argmax(float[][] oneRowMatrix) {
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

    public static int argmin(float[][] oneRowMatrix) {
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

    public static float[][][] append(float[][][] a, float[][][] b) {//Not very general
        int size1 = a.length;
        int size2 = b.length;
        int size3 = size1 + size2;
        float[][][] result = new float[size3][][];
        for (int i = 0; i < size1; i++) {
            result[i] = a[i];
        }
        for (int i = 0; i < size2; i++) {
            result[i + size1] = b[i];
        }
        return result;
    }

    public static float[][] function(float[][] matrix, Function<Float, Float> function) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = function.apply(matrix[i][j]);
            }
        }
        return result;
    }

    public static float[][] bifunction(float[][] m1, float[][] m2, BiFunction<Float, Float, Float> bifunction) {
        int rows = m1.length;
        int columns = m1[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = bifunction.apply(m1[i][j], m2[i][j]);
            }
        }
        return result;
    }
    private static final LinkedList<Function<NN, Stage>> INFOLIST = new LinkedList<>();
    private static final LinkedList<NN> NNLIST = new LinkedList<>();
    private static int FXUpdateRate = 10;

    public static void setFXUpdateRate(int millis) {
        FXUpdateRate = millis;
    }

    public static Function<NN, Stage> infoGraph(boolean mode) {
        return nn -> {
            NumberAxis xAxis = new NumberAxis();
            NumberAxis yAxis = new NumberAxis();
            xAxis.setAnimated(false);
            xAxis.setLabel("Times Backpropagated");
            yAxis.setAnimated(false);
            yAxis.setLabel(mode ? "Accuracy" : "Loss");
            if (mode) {
                yAxis.setAutoRanging(false);
                yAxis.setUpperBound(1);
                yAxis.setTickUnit(.05);
            }
            XYChart.Series<Number, Number> series = new XYChart.Series<>();
            ScatterChart<Number, Number> chart = new ScatterChart<>(xAxis, yAxis);
            chart.setAnimated(false);
            chart.getData().add(series);
            Timeline loop = new Timeline(new KeyFrame(Duration.millis(FXUpdateRate), handler -> {
                if (mode) {
                    series.getData().add(new XYChart.Data<>(nn.sessions, 1 / Math.pow(10, nn.loss)));
                } else {
                    series.getData().add(new XYChart.Data<>(nn.sessions, nn.loss));
                }
            }));
            loop.setCycleCount(-1);
            loop.play();
            Stage stage = new Stage();
            stage.setScene(new Scene(chart, 600, 300));
            return stage;
        };
    }

    public static Function<NN, Stage> infoLayers = nn -> {
        FlowPane network = new FlowPane();
        int size = nn.length;
        Text[] parameters = new Text[size];
        Timeline loop = new Timeline(new KeyFrame(Duration.millis(FXUpdateRate), handler -> {
            for (int i = 0; i < size; i++) {
                parameters[i] = new Text("Layer " + (i + 1) + ":\n" + nn.network[i].toString());
            }
            network.getChildren().clear();
            network.getChildren().addAll(parameters);
        }));
        loop.setCycleCount(-1);
        loop.play();
        Scene scene = new Scene(network);
        Stage stage = new Stage();
        stage.setScene(scene);
        return stage;
    };

    public static void launch() {
        Thread launchThread = new Thread(() -> {
            try {
                launch(NNLib.class);
            } catch (IllegalStateException e) {

            }
        });
        launchThread.setName("NNLib Launch Thread");
        launchThread.start();
    }

    public static void showInfo(Function<NN, Stage> info, NN nn) {
        launch();
        INFOLIST.add(info);
        NNLIST.add(nn);
    }

    @Override
    public void start(Stage stage) {
        Timeline uiUpdateLoop = new Timeline(new KeyFrame(Duration.millis(FXUpdateRate), handler -> {
            if (INFOLIST.size() > 0) {
                Stage infoWindow = INFOLIST.poll().apply(NNLIST.getFirst());
                infoWindow.setTitle(NNLIST.poll().label);
                infoWindow.setX((Screen.getPrimary().getBounds().getWidth() / 5) + (Math.random() * Screen.getPrimary().getBounds().getWidth() / 5));
                infoWindow.show();
            }
        }));
        uiUpdateLoop.setCycleCount(-1);
        uiUpdateLoop.play();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
