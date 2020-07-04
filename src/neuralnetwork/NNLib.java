package neuralnetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.text.Text;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.util.Duration;

public class NNLib extends Application {

    /**
     * A serializable version of java's Function class
     *
     * @param <T> First type
     * @param <R> Second type
     */
    public interface Function<T, R> extends java.util.function.Function<T, R>, Serializable {
    }

    /**
     * A serializable version of java's BiFunction class
     *
     * @param <T> First type
     * @param <S> Second type
     * @param <R> Third type
     */
    public interface BiFunction<T, S, R> extends java.util.function.BiFunction<T, S, R>, Serializable {
    }

    /**
     * Just like java's Function and BiFunction but only with the apply method
     * and 4 parameters.
     *
     * @param <T> First type
     * @param <S> Second type
     * @param <U> Third type
     * @param <V> Fourth type
     * @param <R> Return type
     */
    public interface QuadFunction<T, S, U, V, R> extends Serializable {

        R apply(T t, S s, U u, V v);
    }
    private static int threads;
    private static BiFunction<float[][], float[][], float[][]> dotProduct = (a, b) -> dot(a, b);

    /**
     * The neural network class. This manages all the layers that are put inside
     * the constructor for easy operations. Having a NN instance is optional for
     * a neural network since all the operations can be done with the Layer
     * instances.
     *
     */
    public static class NN implements Serializable {

        private static final long serialVersionUID = 1;
        private Layer[] network;
        private float lr;
        private Random random = new Random();
        private long seed;
        private double loss;
        private long epochs = 0;
        private BiFunction<float[][], float[][], Object[]> lossFunction;
        private QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> optimizer;
        private int step = 1;
        /**
         * The name of this NN instance. Used in saving, loading, and info
         * panels.
         */
        public String label;
        /**
         * The number of layers in the network, not including the input layer.
         */
        public final int length;
        /**
         * The index of the last layer in the network
         */
        public final int lastIndex;

        /**
         * @param label The name for the NN.
         * @param seed A seed for repeatable Layer initialization.
         * @param learningRate A value 0 to 1 for training layer parameters.
         * @param lossFunction Measures the error between two one row matrices.
         * @param optimizer An algorithm that speeds up SGD.
         * @param layers An array or list of layers to be used in the network.
         * @see NNLib.LossFunction
         * @see NNLib.Optimizer
         * @see NNLib.Layer
         */
        public NN(String label, long seed, float learningRate, BiFunction<float[][], float[][], Object[]> lossFunction, QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> optimizer, Layer... layers) {
            this.label = label;
            this.seed = seed;
            lr = learningRate;
            this.lossFunction = lossFunction;
            this.optimizer = optimizer;
            network = layers;
            length = network.length;
            lastIndex = length - 1;
            random.setSeed(seed);
            for (int i = 0; i < length; i++) {
                network[i].initialize(random);
            }
        }

        /**
         * Feed inputs into the network for an output.
         *
         * @param inputs A 2D array to be fed into the network. The shape of the
         * matrix depends on the input layer.
         * @return A 2D array with one 1D array representing the output of the
         * network.
         */
        public float[][] feedforward(float[][] inputs) {
            float[][] out = inputs;
            for (int i = 0; i < length; i++) {
                out = network[i].forward(out);
            }
            return out;
        }

        /**
         * The backpropagation algorithm for tuning weights.
         *
         * @param inputs
         * {@link NN#feedforward(float[][]) Passed into the feedforward method}
         * @param targets The desired output for the network to fit to. Must
         * match the shape of the outputs or either an ArrayOutOfBoundsException
         * will be thrown or some of the targets will be ignored.
         */
        public void backpropagation(float[][] inputs, float[][] targets) {
            float[][] out = feedforward(inputs);
            Object[] lossArr = lossFunction.apply(out, targets);
            loss = (double) lossArr[0];
            float[][] dC_dA = (float[][]) lossArr[1];
            float[][] dC_dZ = network[lastIndex].back(dC_dA, null, lr, optimizer, step);
            for (int i = lastIndex - 1; i >= 0; i--) {
                dC_dZ = network[i].back(dC_dZ, network[i + 1].dZ_dA, lr, optimizer, step);
            }
            step++;
            epochs++;
        }

        /**
         * Randomizes network parameters based on the given range.
         *
         * @param range The range of random values centered at 0.
         */
        public void randomize(float range) {
            for (int i = 0; i < length; i++) {
                network[i].randomize(range);
            }
        }

        /**
         * Adds noise to the network parameters based on the given range.
         *
         * @param range The range of random numbers centered at 0.
         * @param mutateRate Number between 0 and 1 representing the probability
         * that a parameter is altered.
         */
        public void mutate(float range, float mutateRate) {
            for (int i = 0; i < length; i++) {
                network[i].mutate(range, mutateRate);
            }
        }

        @Override
        public String toString() {
            String networkLayers = "Layer 1: " + network[0].toString() + "\n";
            for (int i = 1; i < length; i++) {
                networkLayers += "Layer " + (i + 1) + ": " + network[i].toString() + "\n";
            }
            return networkLayers;
        }

        /**
         * Creates a deep copy of the NN.
         *
         * @return An NN with the same parameters and hyperparameters.
         */
        @Override
        public NN clone() {
            Layer[] layers = new Layer[length];
            for (int i = 0; i < length; i++) {
                layers[i] = network[i].clone();
            }
            NN clone = new NN(label, seed, lr, lossFunction, optimizer, layers);
            return clone;
        }

        /**
         * Copies the parameters from a NN to this NN instance, assuming the
         * network architectures and hyperparameters of both NNs are identical
         * for this to work properly.
         *
         * @param nn The neural network to copy from.
         */
        public void copyParameters(NN nn) {
            for (int i = 0; i < nn.length; i++) {
                network[i] = nn.network[i].clone();
            }
        }

        /**
         * Saves a serialized array of important information of this NN instance
         * into a new file inside the current directory.
         */
        public void save() {
            try {
                FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + File.separator + label + "_neuralnetwork-" + toString());
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                Object[] arr = {network, random, step};
                out.writeObject(arr);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * Same as {@link #load()} method but instead will look outside of the
         * .jar instead of inside.
         *
         * @return True if the load was successful and false if unsuccessful.
         */
        public boolean loadFromJar() {
            try {
                String path = new File(this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath()).getParentFile().getAbsolutePath();//For jar files
                FileInputStream fileIn = new FileInputStream(path + File.separator + label + "_neuralnetwork-" + toString());
                ObjectInputStream in = new ObjectInputStream(fileIn);
                Object[] arr = (Object[]) in.readObject();
                network = (Layer[]) arr[0];
                random = (Random) arr[1];
                step = (Integer) arr[2];
                return true;
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("Could not load network settings for \"" + label + "\".");
                e.printStackTrace();
                return false;
            }
        }

        /**
         * Loads a serialized version of the NN instance created by
         * {@link #save()}. Will search for the file with the same NN label and
         * layer architecture in the name. Careful loading after changing the NN
         * hyper parameters.
         *
         * @return True if the load was successful and false if unsuccessful.
         */
        public boolean load() {
            try {
                FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + File.separator + label + "_neuralnetwork-" + toString());
                ObjectInputStream in = new ObjectInputStream(fileIn);
                Object[] arr = (Object[]) in.readObject();
                network = (Layer[]) arr[0];
                random = (Random) arr[1];
                step = (Integer) arr[2];
                return true;
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("Could not load network settings for \"" + label + "\".");
                e.printStackTrace();
                return false;
            }
        }

        /**
         * Get a specific layer of the NN.
         *
         * @param layerIndex 0 refers to the first hidden layer following the
         * inputs. Passing in one less than the length of the network would
         * return the output layer.
         * @return The corresponding layer.
         */
        public Layer getLayer(int layerIndex) {
            return network[layerIndex];
        }

        /**
         * Get the random class used by the NN with the set seed.
         *
         * @return The random class.
         */
        public Random getRandom() {
            return random;
        }

        /**
         * Get the total amount of times the network has done backpropagation.
         * This number resets everytime the optimizer is changed.
         *
         * @return Times the backpropagation method has been called with the
         * current optimizer.
         */
        public int getSteps() {
            return step;
        }

        /**
         * Change the name of this NN instance.
         *
         * @param label A String to label the network with.
         */
        public void setLabel(String label) {
            this.label = label;
        }

        /**
         * Change the seed of this NN's Random class.
         *
         * @param seed A Long for the seed of the Random class.
         */
        public void setSeed(long seed) {
            this.seed = seed;
            random.setSeed(seed);
        }

        /**
         * Change the learning rate of this NN instance.
         *
         * @param learningRate A double that is casted to a float to avoid
         * having to work with floats.
         */
        public void setLearningRate(float learningRate) {
            lr = learningRate;
        }

        /**
         * Change the loss function of this NN instance.
         *
         * @param lossFunction The loss/cost/error function
         * @see NNLib.LossFunction
         */
        public void setLossFunction(BiFunction<float[][], float[][], Object[]> lossFunction) {
            this.lossFunction = lossFunction;
        }

        /**
         * Change the optimizer of this NN instance.
         *
         * @param optimizer The SGD optimizer
         * @see NNLib.Optimizer
         */
        public void setOptimizer(QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> optimizer) {
            this.optimizer = optimizer;
            step = 1;
        }

        /**
         * Set how many backpropagation steps for gradients to accumulate before
         * tuning parameters. Must be at least 1.
         *
         * @param accumulationSteps The desired amount of steps.
         */
        public void setAccumulationSteps(int accumulationSteps) {
            for (int i = 0; i < length; i++) {
                network[i].setAccumulationSteps(accumulationSteps);
            }
        }
    }

    /**
     * A set of layer types.
     */
    public static abstract class Layer implements Serializable {

        private int accumulationSteps = 1;
        public float[][] dZ_dA;

        /**
         * Set how many backpropagation steps for gradients to accumulate before
         * tuning parameters. Must be at least 1.
         *
         * @param accumulationSteps The desired amount of steps.
         */
        public void setAccumulationSteps(int accumulationSteps) {
            if (accumulationSteps >= 1) {
                this.accumulationSteps = accumulationSteps;
            }
        }

        /**
         * Initializes this Layer's parameters and storage of previous
         * gradients.
         *
         * @param random A Random instance to create random values from.
         */
        public abstract void initialize(Random random);

        /**
         * Feeds values into this Layer and outputs values.
         *
         * @param in Inputs into the network or outputs from the previous Layer.
         * @return Outputs of the Layer.
         */
        public abstract float[][] forward(float[][] in);

        /**
         * Tune the parameters of this Layer with the given parameters. Must
         * give super.dZ_dA a value within the method for Layers behind this one
         * and should be set before making any changes to parameters.
         *
         * @param dC If this Layer is the last in the network, this should be
         * the partial derivatives of the loss with respect to the values after
         * the activation function. Otherwise, this should be the partial
         * derivatives of the loss with respect to the values before the
         * activation function of the Layer in front.
         * @param dZ_dA The partial derivatives of the values before the
         * activation function in the Layer in front with respect to the values
         * of this Layer's values that have been passed into the activation
         * function. Should be null if this Layer is the output layer so the
         * parameters can be properly tuned.
         * @param lr The learning rate for the optimizer.
         * @param optimizer The optimizer to be used to tune the parameters.
         * @param step
         * @return The partial derivatives of the loss with respect to the
         * values before the activation function.
         */
        public abstract float[][] back(float[][] dC, float[][] dZ_dA, float lr, QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> optimizer, int step);

        /**
         * Randomize parameters of this Layer.
         *
         * @param range The range of random values, centered around 0.
         */
        public abstract void randomize(float range);

        /**
         * Adds noise to a portion of the parameters based on the mutate rate.
         *
         * @param range The range of random values, centered around 0.
         * @param mutateRate Value between 0 and 1.
         */
        public abstract void mutate(float range, float mutateRate);

        /**
         * Organizes this Layer's parameters into a String.
         *
         * @return A String that summarizes Layer parameters.
         */
        public abstract String parametersToString();

        /**
         * Creates a deep copy of this Layer.
         *
         * @return A Layer with the same parameters and hyperparameters.
         */
        public abstract Layer clone();

        /**
         * Summarizes the details of this Layer into a String.
         *
         * @return A String with the important specifications of this Layer,
         * leaving out the parameters.
         */
        @Override
        public abstract String toString();

        public static class Dense extends Layer implements Serializable {

            private float[][] weights;
            private float[][] biases;
            private float[][] prevA;
            private float[][] Z;
            private float[][][] updateStorageW;
            private float[][][] updateStorageB;
            private float[][][] accumulated;//First element are for weights, second element are for biases
            private int stepsAccumulated = 0;
            private BiFunction<float[][], Boolean, float[][]> activation;
            private BiFunction<float[][], Integer, float[][]> initializer;
            public final int nodesIn;
            public final int nodesOut;

            public Dense(int nodesIn, int nodesOut, BiFunction<float[][], Boolean, float[][]> activation, BiFunction<float[][], Integer, float[][]> initializer) {
                this.nodesIn = nodesIn;
                this.nodesOut = nodesOut;
                this.activation = activation;
                this.initializer = initializer;
            }

            /**
             * @see #initialize(java.util.Random)
             */
            @Override
            public void initialize(Random random) {
                weights = create(nodesIn, nodesOut, 0);
                biases = create(1, nodesOut, 0);
                weights = NNLib.randomize(weights, 2, -1, random);//values on interval [-1,1]
                biases = NNLib.randomize(biases, 2, -1, random);//values on interval [-1,1]
                weights = initializer.apply(weights, nodesIn);
                updateStorageW = new float[][][]{create(nodesIn, nodesOut, 0)};
                updateStorageB = new float[][][]{create(1, nodesOut, 0)};
                accumulated = new float[2][][];
            }

            /**
             * @see #forward(float[][])
             */
            @Override
            public float[][] forward(float[][] in) {
                prevA = in;
                Z = bifunctionMatrixVectors(dotProduct.apply(in, weights), biases[0], (a, b) -> add(a, b));
                return activation.apply(Z, false);
            }

            /**
             * @see #back(float[][], float[][], float,
             * neuralnetwork.NNLib.QuadFunction, int)
             */
            @Override
            public float[][] back(float[][] dC, float[][] dZ_dA, float lr, QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> optimizer, int step) {
                float[][] dA_dZ = activation.apply(Z, true);
                float[][] dC_dZ;
                if (dZ_dA == null) {//Output Layer
                    if (dC.length == dA_dZ.length && dC[0].length == dA_dZ[0].length) {
                        dC_dZ = multiply(dC, dA_dZ);
                    } else {
                        dC_dZ = dotProduct.apply(dC, dA_dZ);//For jacobian matrices
                    }
                } else {//Hidden Layer
                    float[][] dC_dA = dotProduct.apply(dC, transpose(dZ_dA));
                    if (dC_dA.length == dA_dZ.length && dC_dA[0].length == dA_dZ[0].length) {
                        dC_dZ = multiply(dC_dA, dA_dZ);
                    } else {
                        dC_dZ = dotProduct.apply(dC_dA, dA_dZ);//For jacobian matrices
                    }
                }
                float[][] dC_dW = dotProduct.apply(transpose(prevA), dC_dZ);//prevA = dZ_dW;
                float[][][][] updateW = null;
                while (true) {
                    try {
                        updateW = optimizer.apply(step, lr, dC_dW, updateStorageW);
                        break;
                    } catch (Exception e) {
                        updateStorageW = append(updateStorageW, new float[][][]{create(nodesIn, nodesOut, 0)});
                        updateStorageB = append(updateStorageB, new float[][][]{create(1, nodesOut, 0)});
                    }
                }
                float[][][][] updateB = optimizer.apply(step, lr, dC_dZ, updateStorageB);
                float[][] gradientsW = updateW[0][0];
                float[][] gradientsB = updateB[0][0];
                updateStorageW = updateW[1];
                updateStorageB = updateB[1];
                super.dZ_dA = copy2d(weights);
                tuneParameters(gradientsW, gradientsB);
                return dC_dZ;
            }

            private void tuneParameters(float[][] gradientsW, float[][] gradientsB) {
                if (super.accumulationSteps > 1) {
                    //Add to the accumulated gradients
                    if (accumulated[0] == null) {
                        accumulated[0] = gradientsW;
                        accumulated[1] = gradientsB;
                    } else {
                        accumulated[0] = add(accumulated[0], gradientsW);
                        accumulated[1] = add(accumulated[1], gradientsB);
                    }
                    stepsAccumulated++;
                    if (stepsAccumulated >= super.accumulationSteps) {
                        weights = subtract(weights, accumulated[0]);
                        biases = subtract(biases, accumulated[1]);
                        accumulated = new float[2][][];
                        stepsAccumulated = 0;
                    }
                } else {
                    weights = subtract(weights, gradientsW);
                    biases = subtract(biases, gradientsB);
                }
            }

            /**
             * @see #randomize(float)
             */
            @Override
            public void randomize(float range) {
                weights = NNLib.randomize(weights, range, -range / 2);
                biases = NNLib.randomize(biases, range, -range / 2);
            }

            /**
             * @see #mutate(float, float)
             */
            @Override
            public void mutate(float range, float mutateRate) {
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

            /**
             * @see Layer#parametersToString()
             */
            @Override
            public String parametersToString() {
                return "Weights:\n" + arr2dToString(weights) + "\nBiases:\n" + arr2dToString(biases);
            }

            /**
             * @see Layer#clone()
             */
            @Override
            public Dense clone() {
                Dense copy = new Dense(weights.length, weights[0].length, activation, Initializer.VANILLA);
                copy.weights = copy2d(weights);
                copy.biases = copy2d(biases);
                try {
                    copy.updateStorageW = copy3d(updateStorageW);
                    copy.updateStorageB = copy3d(updateStorageB);
                } catch (Exception e) {
                }
                return copy;
            }

            /**
             * Format of the String: Dense(Nodes In: [nodesIn], Nodes Out:
             * [nodesOut])
             *
             * @see Layer#toString()
             */
            @Override
            public String toString() {
                return "Dense(Nodes In: " + nodesIn + ", Nodes Out: " + nodesOut + ")";
            }
        }
    }

    /**
     * A set of common weight initializers. Takes in weights of a layer and the
     * number of nodes going into the layer ()
     */
    public static class Initializer {

        public static final BiFunction<float[][], Integer, float[][]> VANILLA = (a, b) -> a;//No change
        public static final BiFunction<float[][], Integer, float[][]> XAVIER = (a, b) -> scale(a, (float) Math.sqrt(1.0 / b));
        public static final BiFunction<float[][], Integer, float[][]> HE = (a, b) -> scale(a, (float) Math.sqrt(2.0 / b));
    }

    /**
     * A set of common activation functions. The first parameter is a matrix of
     * values to be passed through a desired function. If the second parameter
     * is true, then the values are passed into the derivative of the desired
     * function.
     */
    public static class ActivationFunction {

        public static final BiFunction<float[][], Boolean, float[][]> LINEAR = (matrix, derivative) -> {
            if (!derivative) {
                return matrix;
            } else {
                float[][] result = create(matrix.length, matrix[0].length, 1);
                return result;
            }
        };
        public static final BiFunction<float[][], Boolean, float[][]> SIGMOID = (matrix, derivative) -> function(matrix, val -> sigmoid(val, derivative));
        public static final BiFunction<float[][], Boolean, float[][]> TANH = (matrix, derivative) -> function(matrix, val -> tanh(val, derivative));
        public static final BiFunction<float[][], Boolean, float[][]> RELU = (matrix, derivative) -> function(matrix, val -> relu(val, derivative));
        public static final BiFunction<float[][], Boolean, float[][]> LEAKYRELU = (matrix, derivative) -> function(matrix, val -> leakyrelu(val, derivative));
        public static final BiFunction<float[][], Boolean, float[][]> SWISH = (matrix, derivative) -> function(matrix, val -> swish(val, derivative));
        public static final BiFunction<float[][], Boolean, float[][]> MISH = (matrix, derivative) -> function(matrix, val -> mish(val, derivative));
        public static final BiFunction<float[][], Boolean, float[][]> SOFTMAX = (matrix, derivative) -> {
            if (!derivative) {
                return softmax(matrix);
            } else {
                int columns = matrix[0].length;
                float[][] softmax = softmax(matrix);
                float[][] jacobian = new float[columns][columns];
                //Diagonal
                for (int i = 0; i < columns; i++) {
                    jacobian[i][i] = softmax[0][i] * (1 - softmax[0][i]);
                    for (int j = 0; j < i; j++) {
                        //Everywhere else are the same on both sides of the diagonal, or symmetric about the diagonal.
                        float val = softmax[0][i] * -softmax[0][j];
                        jacobian[i][j] = val;
                        jacobian[j][i] = val;
                    }
                }
                return jacobian;
            }
        };
    }

    /**
     * A set of common loss functions. First parameter takes in a matrix of
     * outputs from the last layer of the network. The seconds parameter is a
     * matrix of targets with the same shape of the outputs. The return value is
     * an Object array where the first index is the result of the loss function
     * given the outputs and targets and the second index is a matrix with the
     * same shape as the outputs and targets with values from the derivative of
     * the loss function with the outputs as inputs.
     */
    public static class LossFunction {

        /**
         * @param steepnessFactor Recommended value: .5
         * @return Quadratic loss function.
         */
        public static final BiFunction<float[][], float[][], Object[]> QUADRATIC(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                double loss = sum(scale(steepness, square(subtract(outputs, targets)))) / outputs.length;//m(f(x) - y)^2 where f(x) is the output of the network and y is the target output
                return new Object[]{loss, scale(2 * steepness, subtract(outputs, targets))};//Derivative of the loss function for each sample, 2m(f(x) - y)
            };
        }

        /**
         * @param steepnessFactor Recommended value: 1
         * @return Huber loss function.
         */
        public static final BiFunction<float[][], float[][], Object[]> HUBER(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            final float deltaHalf = steepness / 2;
            return (outputs, targets) -> {
                int columns = outputs[0].length;
                float[][] a = subtract(outputs, targets);
                float sum = 0;
                for (int j = 0; j < columns; j++) {
                    float val = a[0][j];
                    if (Math.abs(val) < steepness) {
                        sum += (val * val) / 2;
                    } else {
                        sum += steepness * (Math.abs(a[0][j]) - deltaHalf);
                    }
                }
                double loss = sum;
                float[][] deriv = new float[1][columns];
                for (int j = 0; j < columns; j++) {
                    float val = a[0][j];
                    if (Math.abs(val) < steepness) {
                        deriv[0][j] = a[0][j];
                    } else {
                        deriv[0][j] = steepness * (a[0][j] / Math.abs(a[0][j]));
                    }
                }
                return new Object[]{loss, deriv};
            };
        }

        /**
         * @param steepnessFactor Recommended value: 1
         * @return Pseudo Huber loss function.
         */
        public static final BiFunction<float[][], float[][], Object[]> HUBERPSEUDO(double steepnessFactor) {
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
         * @param steepnessFactor Recommended value: 1
         * @return Cross entropy/log loss function.
         */
        public static final BiFunction<float[][], float[][], Object[]> CROSSENTROPY(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                double loss = steepness * -sum(multiply(targets, ln(outputs)));
                return new Object[]{loss, scale(-steepness, divide(targets, outputs))};
            };
        }
    }

    /**
     * A set of common optimizers. The first three parameters of the
     * QuadFunction are strictly for the time step, learning rate, and the
     * gradients respectively. The third parameter is an array of matrices to
     * store info such as previous updates, gradients, etc. The QuadFunction
     * returns an array where its first element is a array with one element to
     * hold the gradient update matrix and its second element is the storage
     * array that is passed into the next call of the optimizer.
     */
    public static class Optimizer {

        //Should be final?
        public static final float beta = .9f;
        public static final float beta2 = .999f;
        public static final float e = .00000001f;

        private static float[][] EWMA(float beta, float[][] prevStep, float[][] currentStep) {
            return add(scale(beta, prevStep), scale(1 - beta, currentStep));
        }
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> VANILLA = (step, lr, gradients, storage) -> {
            return new float[][][][]{{scale(lr, gradients)}, null};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> MOMENTUM = (step, lr, gradients, storage) -> {
            float[][] update = add(scale(beta, storage[0]), scale(lr, gradients));
            return new float[][][][]{{update}, {update}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> NESTEROV = (step, lr, gradients, storage) -> {//Dozat's modification because calculating two gradients would take a lot of recoding
            float[][] m = add(scale(beta, storage[0]), scale(lr, gradients));
            float[][] update = add(scale(beta, m), scale(lr, gradients));
            return new float[][][][]{{update}, {m}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> ADAGRAD = (step, lr, gradients, storage) -> {
            int rows = gradients.length;
            float val = (1 / (float) Math.sqrt(sum(storage[0]) + e));
            float[][] G = create(rows, rows, 0);
            for (int i = 0; i < rows; i++) {
                G[i][i] = val;
            }
            float[][] update = dotProduct.apply(scale(lr, G), gradients);
            return new float[][][][]{{update}, {square(gradients)}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> ADADELTA = (step, lr, gradients, storage) -> {
            float[][] epsilon = create(gradients.length, gradients[0].length, e);
            float[][] gradientsE = EWMA(beta, storage[0], square(gradients));
            float[][] gradientsRMS = sqrt(add(gradientsE, epsilon));
            float[][] deltaRMS = sqrt(add(storage[1], epsilon));
            float[][] update = multiply(divide(deltaRMS, gradientsRMS), gradients);
            float[][] deltaE = EWMA(beta, storage[1], square(update));
            return new float[][][][]{{update}, {gradientsE, deltaE}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> RMSPROP = (step, lr, gradients, storage) -> {
            float[][] s = EWMA(beta, storage[0], square(gradients));
            float[][] s_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), s);//Bias correction
            float[][] update = divide(scale(lr, gradients), add(sqrt(s_), create(s.length, s[0].length, e)));
            return new float[][][][]{{update}, {s}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> ADAM = (step, lr, gradients, storage) -> {
            float[][] m = EWMA(beta, storage[0], gradients);
            float[][] v = EWMA(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] update = divide(scale(lr, m_), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][][]{{update}, {m, v}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> ADAMAX = (step, lr, gradients, storage) -> {
            float[][] m = EWMA(beta, storage[0], gradients);
            float[][] v = EWMA(beta2, storage[1], square(gradients));
            float[][] m_ = scale(m, 1 / (1 - (float) Math.pow((double) beta, step)));
            float[][] u = max(scale(beta2, storage[1]), abs(gradients));
            float[][] update = divide(scale(lr, m_), add(u, create(u.length, u[0].length, e)));
            return new float[][][][]{{update}, {m, v}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> NADAM = (step, lr, gradients, storage) -> {
            int rows = gradients.length;
            int columns = gradients[0].length;
            float[][] m = EWMA(beta, storage[0], gradients);
            float[][] v = EWMA(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] update = multiply(divide(create(rows, columns, lr), add(sqrt(v_), create(rows, columns, e))), add(scale(beta, m_), scale(scale(1 - beta, gradients), 1 / (1 - beta))));
            return new float[][][][]{{update}, {m, v}};
        };
        public static final QuadFunction<Integer, Float, float[][], float[][][], float[][][][]> AMSGRAD = (step, lr, gradients, storage) -> {
            float[][] m = EWMA(beta, storage[0], gradients);
            float[][] v = EWMA(beta2, storage[1], square(gradients));
            float[][] v_ = max(storage[1], v);
            float[][] update = divide(scale(lr, m), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            return new float[][][][]{{update}, {m, v}};
        };
    }

    public static float[][] normalizeMinMax(float[][] oneRow) {
        int elements = oneRow[0].length;
        float[][] result = new float[1][elements];
        float min = min(oneRow);
        float max = max(oneRow);
        for (int i = 0; i < elements; i++) {
            result[0][i] = (oneRow[0][i] - min) / (max - min);
        }
        return result;
    }

    public static float[][] normalizeZScore(float[][] oneRow) {
        int elements = oneRow[0].length;
        float mean = sum(oneRow) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(oneRow, create(1, elements, mean)))) / (mean)));
        return divide(subtract(oneRow, create(1, elements, mean)), create(1, elements, deviation));
    }

    public static float[][] normalizeTanh(float[][] oneRow) {
        int elements = oneRow[0].length;
        float[][] result = new float[1][elements];
        float mean = sum(oneRow) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(oneRow, create(1, elements, mean)))) / mean));
        for (int i = 0; i < elements; i++) {
            result[0][i] = (.5f * (tanh((.01f * ((oneRow[0][i] - mean) / (deviation))), false) + 1));//Tanh estimator normalization
        }
        return result;
    }

    public static float sigmoid(float x, boolean derivative) {
        if (!derivative) {
            return 1 / (1 + (float) Math.exp(-x));//sigmoid(x)
        } else {
            return sigmoid(x, false) * (1 - sigmoid(x, false));//sigmoid'(x)
        }
    }

    public static float tanh(float x, boolean derivative) {
        if (!derivative) {
            return (2 / (1 + (float) Math.exp(-2 * x))) - 1;//tanh(x)
        } else {
            float val = tanh(x, false);
            return 1 - val * val;//tanh'(x)
        }
    }

    public static float relu(float x, boolean derivative) {
        if (!derivative) {
            return Math.max(0, x);
        } else {
            if (x < 0) {
                return 0;
            }
            return 1;
        }
    }

    public static float leakyrelu(float x, boolean derivative) {
        if (derivative) {
            return Math.max(.001f * x, x);
        } else {
            if (x < 0) {
                return .001f;
            }
            return 1;
        }
    }

    public static float swish(float x, boolean derivative) {
        if (!derivative) {
            return x * sigmoid(x, false);
        } else {
            return x * sigmoid(x, true) + sigmoid(x, false);
        }
    }

    public static float mish(float x, boolean derivative) {
        if (!derivative) {
            return (x * tanh((float) Math.log(1 + Math.exp(x)), false));
        } else {
            double x_ = (double) x;
            return (float) ((Math.exp(x_) * ((4 * (x_ + 1)) + (4 * Math.exp(2 * x_)) + (Math.exp(3 * x_)) + (Math.exp(x_) * (4 * x_ + 6)))) / ((2 * Math.exp(2 * x_)) + (Math.exp(2 * x_)) + 2));
        }
    }

    /**
     * Performs softmax on each of the rows of the given matrix. Includes
     * numeric stability.
     *
     * @param matrix An array holding arrays of values where softmax is
     * performed on each array of values within the array.
     * @return The resulting matrix with the same shape as the given matrix.
     */
    public static float[][] softmax(float[][] matrix) {
        int cols = matrix[0].length;
        float[][] result = matrix;
//        float[][] result = functionMatrixVectors(matrix, vector -> subtract(vector, create(1, cols, max(vector))));//Stabilizing
        result = exp(result);
        return functionMatrixVectors(result, vector -> divide(vector, create(1, cols, sum(vector))));
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

    public static String arr2dToString(float[][] matrix) {
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
        System.out.print(arr2dToString(matrix));
    }

    public static void print(float[][] matrix, String label) {
        System.out.println(label + ":");
        System.out.print(arr2dToString(matrix));
    }

    public static void printDimensions(float[][] matrix) {
        System.out.println("Rows: " + matrix.length + " Columns: " + matrix[0].length);
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
        if (columns1 % 4 == 0) {//Loop unrolling increases speed
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

    public static float[][] inverse(float[][] matrix) {
        return function(matrix, val -> 1 / val);
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

    public static float[][] oneHot(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = create(rows, columns, 0);
        float max = matrix[0][0];
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

    public static float[][] min(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> Math.min(val1, val2));
    }

    public static float[][] max(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> Math.max(val1, val2));
    }

    public static float min(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float min = matrix[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val = matrix[i][j];
                if (val < min) {
                    min = val;
                }
            }
        }
        return min;
    }

    public static float max(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float max = matrix[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val = matrix[i][j];
                if (val > max) {
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

    public static float[] copy1d(float[] arr1d) {
        int length = arr1d.length;
        float[] result = new float[length];
        System.arraycopy(arr1d, 0, result, 0, length);
        return result;
    }

    public static float[][] copy2d(float[][] arr2d) {
        return Arrays.stream(arr2d).map(el -> el.clone()).toArray(a -> arr2d.clone());
    }

    public static float[][][] copy3d(float[][][] arr3d) {
        return Arrays.stream(arr3d).map(m2d -> copy2d(m2d)).toArray(a -> arr3d.clone());
    }

    /**
     * Creates a new array with the elements inside the given arrays in their
     * order. Arrays must have at least 1 element because empty arrays give a
     * length of 1 and the Class of the generic is obtained from the first
     * element from the first array given.
     *
     * @param <T> Object type.
     * @param arrays to be combined
     * @return The new extended array.
     */
    public static <T> T[] append(T[]... arrays) {
        int totalLength = arrays[0].length;
        int numOfArrays = arrays.length;
        for (int i = 1; i < numOfArrays; i++) {
            totalLength += arrays[i].length;
        }
        T[] result = (T[]) Array.newInstance(arrays[0][0].getClass(), totalLength);
        int index = 0;
        for (int i = 0; i < numOfArrays; i++) {
            T[] array = arrays[i];
            int length = array.length;
            for (int j = 0; j < length; j++) {
                result[index] = array[j];
                index++;
            }
        }
        return result;
    }

    public static float[][] functionMatrixVectors(float[][] matrix, Function<float[][], float[][]> operation) {
        int rows = matrix.length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = operation.apply(new float[][]{matrix[i]})[0];
        }
        return result;
    }

    public static float[][] bifunctionMatrixVectors(float[][] matrix, float[] vector, BiFunction<float[][], float[][], float[][]> operation) {
        int rows = matrix.length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = operation.apply(new float[][]{matrix[i]}, new float[][]{vector})[0];
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
    private static int updateRate = 50;
    private static Timeline infoUpdater = new Timeline(new KeyFrame(Duration.millis(100), handler -> {
        if (INFOLIST.size() > 0) {
            Stage infoWindow = INFOLIST.poll().apply(NNLIST.getFirst());
            infoWindow.setTitle(NNLIST.poll().label);
            infoWindow.setX((Screen.getPrimary().getBounds().getWidth() / 5) + (Math.random() * Screen.getPrimary().getBounds().getWidth() / 5));
            infoWindow.show();
        }
    }));
    private static boolean running = false;

    /**
     *
     * @param millis Milliseconds between each update of all info panels.
     * Default is 50 milliseconds
     */
    public static void setInfoUpdateRate(int millis) {
        updateRate = millis;
    }

    private static void updaterBuilder(EventHandler handler) {
        Timeline updater = new Timeline(new KeyFrame(Duration.millis(updateRate), handler));
        updater.setCycleCount(-1);//Indefinite
        updater.play();
    }

    public static Function<NN, Stage> infoGraph(boolean mode) {
        return nn -> {
            NumberAxis xAxis = new NumberAxis();
            NumberAxis yAxis = new NumberAxis();
            xAxis.setAnimated(false);
            xAxis.setLabel("Epochs");
            yAxis.setAnimated(false);
            yAxis.setLabel(mode ? "Accuracy" : "Loss");
            if (mode) {
                yAxis.setForceZeroInRange(false);
            }
            XYChart.Series<Number, Number> series = new XYChart.Series<>();
            ScatterChart<Number, Number> chart = new ScatterChart<>(xAxis, yAxis);
            chart.setAnimated(false);
            chart.getData().add(series);
            updaterBuilder(handler -> {
                if (mode) {
                    series.getData().add(new XYChart.Data<>(nn.epochs, 1 / Math.pow(100 * Math.E, nn.loss)));
                } else {
                    series.getData().add(new XYChart.Data<>(nn.epochs, nn.loss));
                }
            });
            Stage stage = new Stage();
            stage.setScene(new Scene(chart, 600, 300));
            return stage;
        };
    }

    public static Function<NN, Stage> infoLayers = nn -> {
        ScrollPane scroll = new ScrollPane();
        FlowPane network = new FlowPane();
        scroll.setContent(network);
        int size = nn.length;
        Text[] parameters = new Text[size];
        updaterBuilder(handler -> {
            for (int i = 0; i < size; i++) {
                parameters[i] = new Text("Layer " + (i + 1) + ":\n" + nn.network[i].parametersToString());
            }
            network.getChildren().clear();
            network.getChildren().addAll(parameters);
        });
        Scene scene = new Scene(scroll, 600, 300);
        Stage stage = new Stage();
        stage.setScene(scene);
        return stage;
    };

    public static void showInfo(Function<NN, Stage> info, NN nn) {
        if (!running) {
            Thread launchThread = new Thread(() -> {
                try {
                    running = true;
                    launch(NNLib.class);
                } catch (IllegalStateException e) {
                    infoUpdater.setCycleCount(-1);
                    infoUpdater.play();
                    running = true;
                }
            });
            launchThread.setName("NNLib Launch Thread");
            launchThread.start();
        }
        INFOLIST.add(info);
        NNLIST.add(nn);
    }

    @Override
    public void start(Stage stage) {
        infoUpdater.setCycleCount(-1);
        infoUpdater.play();
    }
}
