package nnlibrary;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
import javafx.animation.*;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.text.Text;
import javafx.stage.*;
import javafx.util.Duration;

public class NNlib extends Application {

    /**
     * A serializable version of java's Function interface
     */
    public interface Function<T, R> extends java.util.function.Function<T, R>, Serializable {
    }

    /**
     * A serializable version of java's BiFunction interface
     */
    public interface BiFunction<T, S, R> extends java.util.function.BiFunction<T, S, R>, Serializable {
    }

    private static final long serial = 0;
    private static final int usableProcessors = Runtime.getRuntime().availableProcessors() / 4;
    private static BiFunction<float[][], float[][], float[][]> dotProduct = (a, b) -> {
        if (a.length == 1) {
            return dotVector(a[0], b);
        } else if (a.length / usableProcessors > 10) {
            return dotThreads(a, b, usableProcessors);
        } else {
            return dot(a, b);
        }
    };

    /**
     * The neural network class. This manages all the layers that are put inside
     * the constructor for easy operations. Having a NN instance is optional for
     * a neural network since all the operations can be done with the Layer
     * instances.
     *
     */
    public static class NN implements Serializable {

        private static final long serialVersionUID = serial;
        private Layer[] network;
        private float lr;
        private Random random = new Random();
        private long seed;
        private double loss;
        private double summedLoss = 0;
        private long iterations = 0;
        private LossFunctions.LossFunction lossFunction;
        private Optimizers.Optimizer optimizer;
        private int step = 1;
        private int accumulationSize = 1;
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
         * The index of the last layer in the network.
         */
        public final int lastIndex;

        /**
         * Some variable names do not follow convention so that when the IDE
         * lists out the variables when creating an NN instance you would only
         * have to add a period followed by the name of the already implemented
         * function e.g. Optimizers.ADAM.
         *
         * @param label The name for the NN.
         * @param seed A seed for repeatable Layer initialization.
         * @param learningRate A value 0 to 1 for training layer parameters.
         * @param LossFunctions Measures the error between two one row matrices.
         * @param Optimizers An algorithm that speeds up SGD, or use the VANILLA
         * optimizer for regular SGD.
         * @param Layer The Layer in which inputs first feeds into. Cannot infer
         * input shape since it is the first Layer in the network so it must not
         * be created with the constructor that leaves out the input shape.
         * @param Layers A list of Layers following the input Layer to be used
         * in the network.
         * @see LossFunctions.LossFunction
         * @see Optimizers.Optimizer
         * @see Layer
         */
        public NN(String label, long seed, float learningRate, LossFunctions.LossFunction LossFunctions, Optimizers.Optimizer Optimizers, Layer Layer, Layer... Layers) {
            init(label, seed, learningRate, LossFunctions, Optimizers);
            length = 1 + Layers.length;
            lastIndex = length - 1;
            network = new Layer[length];
            network[0] = Layer;
            Layer.initialize(random);
            for (int i = 1; i < length; i++) {
                Layer current = Layers[i - 1];
                if (current.INFER) {
                    current.initialize(network[i - 1], random);
                } else {
                    current.initialize(random);
                }
                network[i] = current;
            }
        }

        private void init(String label, long seed, float learningRate, LossFunctions.LossFunction lossFunction, Optimizers.Optimizer optimizer) {
            this.label = label;
            this.seed = seed;
            lr = learningRate;
            this.lossFunction = lossFunction;
            this.optimizer = optimizer;
            random.setSeed(seed);
        }

        /**
         * Feed inputs into the network for an output.
         *
         * @param inputs An 2D or 3D array to be fed into the network. The
         * dimensions of the array depends on the input layer.
         * @return A array with the same amount of dimensions as the input
         * representing the output of the network.
         */
        public Object[] feedforward(Object[] inputs) {
            Object[] out = inputs;
            for (int i = 0; i < length; i++) {
                out = network[i].forward(out);
            }
            return out;
        }

        /**
         * The backpropagation algorithm for tuning weights.
         *
         * @param inputs
         * {@link NN#feedforward(java.lang.Object[]) Passed into the feedforward method}
         * @param targets The desired output for the network to fit to. Must
         * match the shape of the outputs or either an ArrayOutOfBoundsException
         * will be thrown or some of the targets will be ignored.
         */
        public void backpropagation(Object[] inputs, Object[] targets) {
            boolean update;
            if (iterations % accumulationSize == 0) {
                update = true;
            } else {
                update = false;
            }
            float[][] out = (float[][]) feedforward(inputs);
            Object[] lossArr = lossFunction.apply(out, (float[][]) targets);
            summedLoss += (double) lossArr[0];
            Object[] dC_dA = (Object[]) lossArr[1];
            dC_dA = network[lastIndex].back(dC_dA, lr, optimizer, step, update);
            for (int i = lastIndex - 1; i >= 0; i--) {
                dC_dA = network[i].back(dC_dA, lr, optimizer, step, update);
            }
            if (update) {
                step++;
                loss = summedLoss / accumulationSize;
                summedLoss = 0;
            }
            iterations++;
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
            String networkLayers = network[0].toString();
            for (int i = 1; i < length; i++) {
                networkLayers += network[i].toString();
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
            Layer inputLayer = network[0].clone();
            Layer[] layers = new Layer[lastIndex];
            for (int i = 1; i < length; i++) {
                layers[i - 1] = network[i].clone();
            }
            NN clone = new NN(label, seed, lr, lossFunction, optimizer, inputLayer, layers);
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
         * Get the total number of adjustable(through backpropagation)
         * parameters in this NN.
         *
         * @return An integer representing the total amount of adjustable
         * parameters.
         */
        public int getParameterCount() {
            int count = network[0].getParameterCount();
            for (int i = 1; i < length; i++) {
                count += network[i].getParameterCount();
            }
            return count;
        }

        /**
         * Saves a serialized array of important information of this NN instance
         * into a new file inside the current directory.
         */
        public void save() {
            try {
                FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + File.separator + label + "-" + toString());
                ObjectOutputStream out = new ObjectOutputStream(fileOut);
                Object[] arr = {network, random, step};
                out.writeObject(arr);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * Saves a serialized array of important information of this NN instance
         * into a new file inside the current directory. Works for both .jar and
         * while uncompiled, will save in the .jar root or the src folder if
         * uncompiled.
         */
        public void saveInsideJar() {
            try {
                String root = this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath();
                if (!root.substring(root.length() - 4).equals(".jar")) {
                    FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + File.separator + "src" + File.separator + label + "-" + toString());
                    ObjectOutputStream out = new ObjectOutputStream(fileOut);
                    Object[] arr = {network, random, step};
                    out.writeObject(arr);
                } else {
                    FileOutputStream fileOut = new FileOutputStream(root);
                    ObjectOutputStream out = new ObjectOutputStream(fileOut);
                    Object[] arr = {network, random, step};
                    out.writeObject(arr);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * Loads a serialized NN instance created by {@link #save()}. Will
         * search for the file with the same NN label and layer architecture in
         * the name. Careful loading after changing the NN hyper parameters. The
         * directory is in the current user folder if used as a jar executable
         *
         * @return True if the load was successful and false if unsuccessful.
         */
        public boolean load() {
            try {
                FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + File.separator + label + "-" + toString());
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
         * Same as {@link #load()} method but instead will look outside of the
         * .jar.
         *
         * @return True if the load was successful and false if unsuccessful.
         */
        public boolean loadOutsideJar() {
            try {
                String path = new File(this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath()).getParentFile().getAbsolutePath();//For jar files
                FileInputStream fileIn = new FileInputStream(path + File.separator + label + "-" + toString());
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
         * Same as {@link #load()} method but instead will look inside of the
         * .jar.
         *
         * @return True if the load was successful and false if unsuccessful.
         */
        public boolean loadInsideJar() {
            try {
                String root = this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath();
                if (root.substring(root.length() - 4).equals(".jar")) {
                    InputStream stream = this.getClass().getResource("/" + label + "-" + toString()).openStream();
                    ObjectInputStream in = new ObjectInputStream(stream);
                    Object[] arr = (Object[]) in.readObject();
                    network = (Layer[]) arr[0];
                    random = (Random) arr[1];
                    step = (Integer) arr[2];
                    return true;
                } else {
                    FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + File.separator + "src" + File.separator + label + "-" + toString());
                    ObjectInputStream in = new ObjectInputStream(fileIn);
                    Object[] arr = (Object[]) in.readObject();
                    network = (Layer[]) arr[0];
                    random = (Random) arr[1];
                    step = (Integer) arr[2];
                    return true;
                }
            } catch (IOException | ClassNotFoundException | NullPointerException e) {
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
         * Get the seed of the Random class being used by this NN instance.
         *
         * @return A long representing the seed.
         */
        public long getSeed() {
            return seed;
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
         * @see LossFunctions.LossFunction
         */
        public void setLossFunction(LossFunctions.LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }

        /**
         * Change the optimizer of this NN instance. WIP
         *
         * @param optimizer The SGD optimizer
         * @see Optimizers.Optimizer
         */
        public void setOptimizer(Optimizers.Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        /**
         * Set how many backpropagation steps for gradients to accumulate before
         * tuning parameters. Must be at least 1.
         *
         * @param size The desired amount of steps.
         */
        public void setAccumulationSize(int size) {
            if (size >= 1) {
                this.accumulationSize = size;
            }
        }
    }

    /**
     * A set of layer types.
     */
    public static abstract class Layer implements Serializable {

        private final boolean INFER;
        private int[] OUTSHAPE;

        private Layer(boolean infer) {
            INFER = infer;
        }

        /**
         * Initializes this Layer's parameters and storage of previous
         * gradients.
         *
         * @param random A Random instance to create random values from.
         */
        public abstract void initialize(Random random);

        /**
         * Initializes this Layer's parameters and storage of previous
         * gradients. With inference of input shape from the prior layer.
         *
         * @param previousLayer The prior layer to infer input shape from.
         * @param random A Random instance to create random values from.
         */
        public abstract void initialize(Layer previousLayer, Random random);

        /**
         * Feeds values into this Layer and outputs values.
         *
         * @param in Inputs into the network or outputs from the previous Layer.
         * @return Outputs of the Layer.
         */
        public abstract Object[] forward(Object[] in);

        /**
         * Tune the parameters of this Layer with the given parameters.
         *
         * @param dC_dA_uncasted The partial derivatives of the loss with
         * respect to the activations of the current Layer.
         * @param lr The learning rate for the optimizer.
         * @param optimizer The optimizer to be used to tune the parameters.
         * @param step The number of times the parameters have been tuned. This
         * can be set to 0 but will cause biased gradients.
         * @param update Whether or not to tune the parameters after computing
         * gradients.
         * @return The partial derivatives of the loss with respect to the
         * values before the activation function.
         */
        public abstract Object[] back(Object[] dC_dA_uncasted, float lr, Optimizers.Optimizer optimizer, int step, boolean update);

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
         * Get the number of learnable parameters in this Layer.
         *
         * @return An int representing the total learnable parameters.
         */
        public abstract int getParameterCount();

        /**
         * Creates a deep copy of this Layer.
         *
         * @return A Layer with the same parameters and hyperparameters.
         */
        public abstract Layer clone();

        /**
         * Summarizes the details of this Layer into a String. Needed in the
         * save method in the NN class, so it must leave out characters that
         * cannot be used in a file name.
         *
         * @return A String with the important specifications of this Layer,
         * leaving out the parameters.
         */
        @Override
        public abstract String toString();

        public static class Dense extends Layer implements Serializable {

            private static final long serialVersionUID = serial;
            private float[][] weights;
            private float[][] biases;
            private float[][] prevA;
            private float[][] Z;
            private float[][][] updateStorageW;
            private float[][][] updateStorageB;
            private float[][] accumulatedW;
            private float[][] accumulatedB;
            private Activations.Activation activation;
            private Initializers.Initializer initializer;
            private int nodesIn;
            private final int nodesOut;

            public Dense(int nodesIn, int nodesOut, Activations.Activation activation, Initializers.Initializer initializer) {
                super(false);
                this.nodesIn = nodesIn;
                this.nodesOut = nodesOut;
                this.activation = activation;
                this.initializer = initializer;
            }

            public Dense(int nodesOut, Activations.Activation activation, Initializers.Initializer initializer) {
                super(true);
                this.nodesOut = nodesOut;
                this.activation = activation;
                this.initializer = initializer;
            }

            /**
             * @see Layer#initialize(java.util.Random)
             */
            @Override
            public void initialize(Random random) {
                weights = NNlib.randomize(new float[nodesIn][nodesOut], 2, -1, random);//values on interval [-1,1]
                biases = NNlib.randomize(new float[1][nodesOut], 2, -1, random);//values on interval [-1,1]
                weights = initializer.apply(weights, nodesIn);
                updateStorageW = new float[1][nodesIn][nodesOut];
                updateStorageB = new float[1][1][nodesOut];
                accumulatedW = new float[nodesIn][nodesOut];
                accumulatedB = new float[1][nodesOut];
                super.OUTSHAPE = new int[]{nodesOut};
            }

            /**
             * @see Layer#initialize(nnlibrary.NNlib.Layer, java.util.Random)
             */
            @Override
            public void initialize(Layer previousLayer, Random random) {
                nodesIn = previousLayer.OUTSHAPE[0];
                weights = NNlib.randomize(new float[nodesIn][nodesOut], 2, -1, random);//values on interval [-1,1]
                biases = NNlib.randomize(new float[1][nodesOut], 2, -1, random);//values on interval [-1,1]
                weights = initializer.apply(weights, nodesIn);
                updateStorageW = new float[1][nodesIn][nodesOut];
                updateStorageB = new float[1][1][nodesOut];
                accumulatedW = new float[nodesIn][nodesOut];
                accumulatedB = new float[1][nodesOut];
                super.OUTSHAPE = new int[]{nodesOut};
            }

            /**
             * @see Layer#forward(java.lang.Object[])
             */
            @Override
            public Object[] forward(Object[] in) {
                prevA = (float[][]) in;
                Z = add(dotProduct.apply(prevA, weights), biases);
                return activation.apply(Z, false);
            }

            /**
             * @see Layer#back(java.lang.Object[], float,
             * nnlibrary.NNlib.Optimizers.Optimizer, int, boolean)
             */
            @Override
            public Object[] back(Object[] dC_dA_uncasted, float lr, Optimizers.Optimizer optimizer, int step, boolean update) {
                float[][] dC_dA = (float[][]) dC_dA_uncasted;
                float[][] dA_dZ = activation.apply(Z, true);
                float[][] dC_dZ;
                if (dC_dA.length == dA_dZ.length && dC_dA[0].length == dA_dZ[0].length) {
                    dC_dZ = multiply(dC_dA, dA_dZ);
                } else {
                    dC_dZ = dotProduct.apply(dC_dA, dA_dZ);//For jacobian matrices
                }
                float[][] dC_dW = dotProduct.apply(transpose(prevA), dC_dZ);//prevA = dZ_dW;
                Object[] updateW;
                while (true) {
                    try {
                        updateW = optimizer.apply(step, lr, dC_dW, updateStorageW);
                        break;
                    } catch (Exception e) {
                        updateStorageW = append(updateStorageW, new float[1][nodesIn][nodesOut]);
                        updateStorageB = append(updateStorageB, new float[1][1][nodesOut]);
                    }
                }
                Object[] updateB = optimizer.apply(step, lr, dC_dZ, updateStorageB);
                updateStorageW = (float[][][]) updateW[1];
                updateStorageB = (float[][][]) updateB[1];
                float[][] dC_dA_ = dotProduct.apply(dC_dZ, transpose(weights));
                tuneParameters((float[][]) updateW[0], (float[][]) updateB[0], update);
                return dC_dA_;
            }

            public void tuneParameters(float[][] gradientsW, float[][] gradientsB, boolean update) {
                //Add to the accumulated gradients
                accumulatedW = add(accumulatedW, gradientsW);
                accumulatedB = add(accumulatedB, gradientsB);
                if (update) {
                    weights = subtract(weights, accumulatedW);
                    biases = subtract(biases, accumulatedB);
                    accumulatedW = new float[nodesIn][nodesOut];
                    accumulatedB = new float[1][nodesOut];
                }
            }

            /**
             * @see #randomize(float)
             */
            @Override
            public void randomize(float range) {
                weights = NNlib.randomize(weights, range, -range / 2);
                biases = NNlib.randomize(biases, range, -range / 2);
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
                return "Weights:\n" + arrToString(weights) + "\nBiases:\n" + arrToString(biases);
            }

            /**
             * @see Layer#getParameterCount()
             */
            @Override
            public int getParameterCount() {
                return nodesIn * nodesOut + nodesOut;
            }

            /**
             * @see Layer#clone()
             */
            @Override
            public Dense clone() {
                Dense copy = new Dense(weights.length, weights[0].length, activation, initializer);
                copy.weights = copy(weights);
                copy.biases = copy(biases);
                try {
                    copy.updateStorageW = copy(updateStorageW);
                    copy.updateStorageB = copy(updateStorageB);
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
                return "Dense[" + nodesIn + "_" + nodesOut + "]";
            }
        }

        public static class Conv extends Layer implements Serializable {

            private static final long serialVersionUID = serial;
            private int inputHeight;
            private int inputWidth;
            private float[][][][] filters;
            private final int filterNum;
            private int filterChannels;
            private final int filterHeight;
            private final int filterWidth;
            private float[] biases;
            private final int stride;
            private final int paddingHeight;
            private final int paddingWidth;
            private final Function<float[][], float[][]> pad;
            private final Function<float[][], float[][]> unpad;
            private Activations.Activation activation;
            private float[][][][] updateStorageF;
            private float[][][] updateStorageB;
            private float[][][][] accumulatedF;
            private float[] accumulatedB;
            private float[][][] prevA;
            private float[][][] Z;

            public Conv(int inputChannels, int inputHeight, int inputWidth, int numberOfFilters, int filterHeight, int filterWidth, int stride, int paddingHeight, int paddingWidth, Activations.Activation activation) {
                super(false);
                this.inputHeight = inputHeight;
                this.inputWidth = inputWidth;
                filterNum = numberOfFilters;
                this.filterChannels = inputChannels;
                this.filterHeight = filterHeight;
                this.filterWidth = filterWidth;
                this.stride = stride;
                this.paddingHeight = paddingHeight;
                this.paddingWidth = paddingWidth;
                this.activation = activation;
                pad = a -> NNlib.pad(a, paddingHeight, paddingWidth);
                unpad = a -> NNlib.unpad(a, paddingHeight, paddingWidth);
            }

            public Conv(int numberOfFilters, int filterHeight, int filterWidth, int stride, int paddingHeight, int paddingWidth, Activations.Activation activation) {
                super(true);
                filterNum = numberOfFilters;
                this.filterHeight = filterHeight;
                this.filterWidth = filterWidth;
                this.stride = stride;
                this.paddingHeight = paddingHeight;
                this.paddingWidth = paddingWidth;
                this.activation = activation;
                pad = a -> NNlib.pad(a, paddingHeight, paddingWidth);
                unpad = a -> NNlib.unpad(a, paddingHeight, paddingWidth);
            }

            /**
             * @see Layer#initialize(java.util.Random)
             */
            @Override
            public void initialize(Random random) {
                filters = new float[filterNum][filterChannels][][];
                biases = new float[filterNum];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        filters[i][j] = NNlib.randomize(new float[filterHeight][filterWidth], 2, -1, random);
                    }
                    biases[i] = random.nextFloat();
                }
                updateStorageF = new float[filterNum * filterChannels][1][filterHeight][filterWidth];
                updateStorageB = new float[1][1][filterNum];
                accumulatedF = new float[filterNum][filterChannels][filterHeight][filterWidth];
                accumulatedB = new float[filterNum];
                super.OUTSHAPE = new int[]{filterNum, (inputHeight + 2 * paddingHeight - filterHeight) / stride + 1, (inputWidth + 2 * paddingWidth - filterWidth) / stride + 1};
            }

            /**
             * @see Layer#initialize(nnlibrary.NNlib.Layer, java.util.Random)
             */
            @Override
            public void initialize(Layer previousLayer, Random random) {
                int[] shapeIn = previousLayer.OUTSHAPE;
                filterChannels = shapeIn[0];
                inputHeight = shapeIn[1];
                inputWidth = shapeIn[2];
                filters = new float[filterNum][filterChannels][][];
                biases = new float[filterNum];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        filters[i][j] = NNlib.randomize(new float[filterHeight][filterWidth], 2, -1, random);
                    }
                    biases[i] = random.nextFloat();
                }
                updateStorageF = new float[filterNum * filterChannels][1][filterHeight][filterWidth];
                updateStorageB = new float[1][1][filterNum];
                accumulatedF = new float[filterNum][filterChannels][filterHeight][filterWidth];
                accumulatedB = new float[filterNum];
                super.OUTSHAPE = new int[]{filterNum, (inputHeight + 2 * paddingHeight - filterHeight) / stride + 1, (inputWidth + 2 * paddingWidth - filterWidth) / stride + 1};
            }

            /**
             * @see Layer#forward(java.lang.Object[])
             */
            @Override
            public Object[] forward(Object[] in) {
                try {
                    prevA = function2dOn3d((float[][][]) in, pad);
                    Z = NNlib.convolution(prevA, filters, biases, stride);
                    return function2dOn3d(Z, a -> activation.apply(a, false));
                } catch (ArrayIndexOutOfBoundsException e) {
                    throw new IllegalArgumentException("Input dimensions are " + getDimensions(in) + " but the filter size is " + getDimensions(filters));
                }
            }

            /**
             * @see Layer#back(java.lang.Object[], float,
             * nnlibrary.NNlib.Optimizers.Optimizer, int, boolean)
             */
            @Override
            public Object[] back(Object[] dC_dA_uncasted, float lr, Optimizers.Optimizer optimizer, int step, boolean update) {
                float[][][] dC_dA = (float[][][]) dC_dA_uncasted;
                float[][][] dA_dZ = function2dOn3d(Z, a -> activation.apply(a, true));
                float[][][] dC_dZ;
                if (dC_dA[0].length == dA_dZ[0].length && dC_dA[0][0].length == dA_dZ[0][0].length) {
                    dC_dZ = bifunction2dOn3d(dC_dA, dA_dZ, (a, b) -> multiply(a, b));
                } else {
                    dC_dZ = bifunction2dOn3d(dC_dA, dA_dZ, (a, b) -> dotProduct.apply(a, b));
                }
                dC_dZ = function2dOn3d(dC_dZ, a -> dilate(a, stride - 1, stride - 1));//For strides > 1
                float[][][][] dC_dW = new float[filterNum][filterChannels][][];
                float[] dC_dB = new float[filterNum];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        dC_dW[i][j] = convolution(new float[][][]{prevA[j]}, new float[][][][]{{dC_dZ[i]}}, new float[]{0}, 1)[0];
                    }
                    dC_dB[i] = sum(dC_dZ[i]);
                }
                float[][][][] updateF = new float[filterNum][filterChannels][][];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        Object[] optimizedF;
                        int index = i * filterChannels + j;
                        while (true) {
                            try {
                                optimizedF = optimizer.apply(step, lr, dC_dW[i][j], updateStorageF[index]);
                                break;
                            } catch (ArrayIndexOutOfBoundsException e) {
//                                e.printStackTrace();
                                updateStorageF[index] = append(updateStorageF[index], new float[1][filterHeight][filterWidth]);
                                updateStorageB = append(updateStorageB, new float[1][1][filterNum]);
                            }
                        }
                        updateF[i][j] = (float[][]) optimizedF[0];
                        updateStorageF[i] = (float[][][]) optimizedF[1];
                    }
                }
                Object[] optimizedB = optimizer.apply(step, lr, new float[][]{dC_dB}, updateStorageB);
                float[][] updateB = (float[][]) optimizedB[0];
                updateStorageB = (float[][][]) optimizedB[1];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        accumulatedF[i][j] = add(accumulatedF[i][j], updateF[i][j]);
                    }
                    accumulatedB[i] += updateB[0][i];
                }
                if (update) {
                    for (int i = 0; i < filterNum; i++) {
                        for (int j = 0; j < filterChannels; j++) {
                            filters[i][j] = subtract(filters[i][j], accumulatedF[i][j]);
                        }
                        biases[i] -= accumulatedB[i];
                    }
                    accumulatedF = new float[filterNum][filterChannels][filterHeight][filterWidth];
                    accumulatedB = new float[filterNum];
                }
                float[][][][] rotatedFilters = new float[filterNum][filterChannels][][];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        rotatedFilters[i][j] = rotate180(filters[i][j]);
                    }
                }
                float[][][] dC_dZ_dilated_padded = function2dOn3d(dC_dZ, a -> pad(a, filterHeight - 1, filterWidth - 1));//Full convolution
                float[][][] dC_dA_ = new float[filterChannels][prevA[0].length][prevA[0][0].length];
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        dC_dA_[j] = add(dC_dA_[j], convolution(new float[][][]{dC_dZ_dilated_padded[i]}, new float[][][][]{{rotatedFilters[i][j]}}, new float[filterNum], 1)[0]);
                    }
                }
                dC_dA_ = function2dOn3d(dC_dA_, unpad);
                return dC_dA_;
            }

            /**
             * @see Layer#randomize(float)
             */
            @Override
            public void randomize(float range) {
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        NNlib.randomize(filters[i][j], range, -range / 2);
                    }
                    biases[i] = (float) Math.random() * range + -range / 2;
                }
            }

            /**
             * @see Layer#mutate(float, float)
             */
            @Override
            public void mutate(float range, float mutateRate) {
                for (int i = 0; i < filterNum; i++) {
                    for (int j = 0; j < filterChannels; j++) {
                        for (int k = 0; k < filterHeight; k++) {
                            for (int l = 0; l < filterWidth; l++) {
                                if (Math.random() < mutateRate) {
                                    filters[i][j][k][l] = (float) Math.random() * range + -range / 2;
                                }
                            }
                        }
                    }
                    if (Math.random() < mutateRate) {
                        biases[i] = (float) Math.random() * range + -range / 2;
                    }
                }
            }

            /**
             * @see Layer#parametersToString()
             */
            @Override
            public String parametersToString() {
                String parameters = "";
                for (int i = 0; i < filterNum; i++) {
                    String currentFilter = "Filter " + (i + 1);
                    parameters += currentFilter + ": \n" + arrToString(filters[i]) + currentFilter + " Bias: [" + biases[i] + "]\n";
                }
                return parameters;
            }

            /**
             * @see Layer#getParameterCount()
             */
            @Override
            public int getParameterCount() {
                return filterNum * filterChannels * filterHeight * filterWidth + filterNum;
            }

            /**
             * @see Layer#clone()
             */
            @Override
            public Layer clone() {
                Conv copy = new Conv(filterChannels, inputHeight, inputWidth, filterNum, filterHeight, filterWidth, stride, paddingHeight, paddingWidth, activation);
                copy.filters = copy(filters);
                copy.biases = copy(biases);
                copy.updateStorageF = copy(updateStorageF);
                copy.updateStorageB = copy(updateStorageB);
                return copy;
            }

            /**
             * @see Layer#toString()
             */
            @Override
            public String toString() {
                return "Conv[" + filterNum + "_" + filterChannels + "_" + filterHeight + "_" + filterWidth + "_" + stride + "]";
            }
        }

        public static class Flatten extends Layer implements Serializable {

            private static final long serialVersionUID = serial;
            private int channels;
            private int height;
            private int width;

            public Flatten(int channelsIn, int heightIn, int widthIn) {
                super(false);
                channels = channelsIn;
                height = heightIn;
                width = widthIn;
            }

            public Flatten() {
                super(true);
            }

            /**
             * @see Layer#initialize(java.util.Random)
             */
            @Override
            public void initialize(Random random) {
                super.OUTSHAPE = new int[]{channels * height * width};
            }

            /**
             * @see Layer#initialize(nnlibrary.NNlib.Layer, java.util.Random)
             */
            @Override
            public void initialize(Layer previousLayer, Random random) {
                int[] shapeIn = previousLayer.OUTSHAPE;
                channels = shapeIn[0];
                height = shapeIn[1];
                width = shapeIn[2];
                super.OUTSHAPE = new int[]{channels * height * width};
            }

            /**
             * @see Layer#forward(java.lang.Object[])
             */
            @Override
            public Object[] forward(Object[] in) {
                float[][][] convOut = (float[][][]) in;
                float[][] flattened = new float[1][channels * height * width];
                int index = 0;
                for (int i = 0; i < channels; i++) {
                    for (int j = 0; j < height; j++) {
                        for (int k = 0; k < width; k++) {
                            flattened[0][index] = convOut[i][j][k];
                            index++;
                        }
                    }
                }
                return flattened;
            }

            /**
             * @see Layer#back(java.lang.Object[], float,
             * nnlibrary.NNlib.Optimizers.Optimizer, int, boolean)
             */
            @Override
            public Object[] back(Object[] dC_dA_uncasted, float lr, Optimizers.Optimizer optimizer, int step, boolean update) {
                float[][] dC_dA = (float[][]) dC_dA_uncasted;
                float[][][] unflattened = new float[channels][height][width];
                int index = 0;
                for (int i = 0; i < channels; i++) {
                    for (int j = 0; j < height; j++) {
                        for (int k = 0; k < width; k++) {
                            unflattened[i][j][k] = dC_dA[0][index];
                            index++;
                        }
                    }
                }
                return unflattened;
            }

            /**
             * @see Layer#randomize(float)
             */
            @Override
            public void randomize(float range) {
            }

            /**
             * @see Layer#mutate(float, float)
             */
            @Override
            public void mutate(float range, float mutateRate) {
            }

            /**
             * @see Layer#parametersToString()
             */
            @Override
            public String parametersToString() {
                return "Flatten Layer";
            }

            /**
             * @see Layer#getParameterCount()
             */
            @Override
            public int getParameterCount() {
                return 0;
            }

            /**
             * @see Layer#clone()
             */
            @Override
            public Layer clone() {
                return new Flatten(channels, height, width);
            }

            /**
             * @see Layer#toString()
             */
            @Override
            public String toString() {
                return "Flat[]";
            }
        }

        public static class Maxpool extends Layer implements Serializable {

            private static final long serialVersionUID = serial;
            private int inputChannels;
            private int inputHeight;
            private int inputWidth;
            private final int poolingHeight;
            private final int poolingWidth;
            private final int stride;
            private int resultHeight;
            private int resultWidth;
            private int[][] positions;

            public Maxpool(int inputChannels, int inputHeight, int inputWidth, int poolingHeight, int poolingWidth, int stride) {
                super(false);
                this.inputChannels = inputChannels;
                this.inputHeight = inputHeight;
                this.inputWidth = inputWidth;
                this.poolingHeight = poolingHeight;
                this.poolingWidth = poolingWidth;
                this.stride = stride;
                resultHeight = (inputHeight - poolingHeight) / stride + 1;
                resultWidth = (inputWidth - poolingWidth) / stride + 1;
                positions = new int[inputChannels * resultHeight * resultWidth][2];
            }

            public Maxpool(int poolingHeight, int poolingWidth, int stride) {
                super(true);
                this.poolingHeight = poolingHeight;
                this.poolingWidth = poolingWidth;
                this.stride = stride;
            }

            /**
             * @see Layer#initialize(java.util.Random)
             */
            @Override
            public void initialize(Random random) {
                super.OUTSHAPE = new int[]{inputChannels, resultHeight, resultWidth};
            }

            /**
             * @see Layer#initialize(nnlibrary.NNlib.Layer, java.util.Random)
             */
            @Override
            public void initialize(Layer previousLayer, Random random) {
                int[] shapeIn = previousLayer.OUTSHAPE;
                inputChannels = shapeIn[0];
                inputHeight = shapeIn[1];
                inputWidth = shapeIn[2];
                resultHeight = (inputHeight - poolingHeight) / stride + 1;
                resultWidth = (inputWidth - poolingWidth) / stride + 1;
                positions = new int[inputChannels * resultHeight * resultWidth][2];
                super.OUTSHAPE = new int[]{inputChannels, resultHeight, resultWidth};
            }

            /**
             * @see Layer#forward(java.lang.Object[])
             */
            @Override
            public Object[] forward(Object[] in) {
                try {
                    float[][][] arr3d = (float[][][]) in;
                    float[][][] pooled = new float[inputChannels][resultHeight][resultWidth];
                    for (int rDepth = 0; rDepth < inputChannels; rDepth++) {
                        float[][] pool2d = pooled[rDepth];
                        float[][] arr2d = arr3d[rDepth];
                        for (int rRow = 0; rRow < resultHeight; rRow++) {
                            int startRow = rRow * stride;
                            int boundRow = startRow + poolingHeight;
                            for (int rCol = 0; rCol < resultWidth; rCol++) {
                                int startCol = rCol * stride;
                                int boundCol = startCol + poolingWidth;
                                float max = Float.NEGATIVE_INFINITY;
                                int row = 0;
                                int col = 0;
                                for (int iRow = startRow; iRow < boundRow; iRow++) {
                                    for (int iCol = startCol; iCol < boundCol; iCol++) {
                                        float val = arr2d[iRow][iCol];
                                        if (val > max) {
                                            max = val;
                                            row = iRow;
                                            col = iCol;
                                        }
                                    }
                                }
                                int[] position = positions[rDepth * resultHeight + rRow * resultWidth + rCol];
                                pool2d[rRow][rCol] = max;
                                position[0] = row;
                                position[1] = col;
                            }
                        }
                    }
                    return pooled;
                } catch (ArrayIndexOutOfBoundsException e) {
                    throw new IllegalArgumentException("Input dimensions are " + getDimensions(in) + " but should be " + inputChannels + " " + inputHeight + " " + inputWidth);
                }
            }

            /**
             * @see Layer#back(java.lang.Object[], float,
             * nnlibrary.NNlib.Optimizers.Optimizer, int, boolean)
             */
            @Override
            public Object[] back(Object[] dC_dA_uncasted, float lr, Optimizers.Optimizer optimizer, int step, boolean update) {
                float[][][] dC_dA = (float[][][]) dC_dA_uncasted;
                float[][][] unpooled = new float[inputChannels][inputHeight][inputWidth];
                for (int i = 0; i < inputChannels; i++) {
                    for (int j = 0; j < resultHeight; j++) {
                        for (int k = 0; k < resultWidth; k++) {
                            int[] position = positions[i * resultHeight + j * resultWidth + k];
                            unpooled[i][position[0]][position[1]] += dC_dA[i][j][k];
                        }
                    }
                }
                return unpooled;
            }

            /**
             * @see Layer#randomize(float)
             */
            @Override
            public void randomize(float range) {
            }

            /**
             * @see Layer#mutate(float, float)
             */
            @Override
            public void mutate(float range, float mutateRate) {
            }

            /**
             * @see Layer#parametersToString()
             */
            @Override
            public String parametersToString() {
                return "Maxpool Layer";
            }

            /**
             * @see Layer#getParameterCount()
             */
            @Override
            public int getParameterCount() {
                return 0;
            }

            /**
             * @see Layer#clone()
             */
            @Override
            public Layer clone() {
                return new Maxpool(inputChannels, inputHeight, inputWidth, poolingHeight, poolingWidth, stride);
            }

            /**
             * @see Layer#toString()
             */
            @Override
            public String toString() {
                return "MPool[]";
            }
        }
    }

    /**
     * A set of common weight initializers. Takes in weights of a layer and the
     * number of nodes going into the layer ()
     */
    public static class Initializers {

        /**
         * A functional interface for the initialization of parameters.
         */
        public interface Initializer extends Serializable {

            /**
             * Applies the initializer to the given parameters.
             *
             * @param parameters A 2d array of parameters for the initializer to
             * modify.
             * @param nodesIn The nodes going into the Layer with the
             * parameters.
             * @return A 2d array with the new parameters.
             */
            float[][] apply(float[][] parameters, int nodesIn);
        }
        public static final Initializer VANILLA = (a, b) -> a;//No change
        public static final Initializer XAVIER = (a, b) -> multiply(a, (float) Math.sqrt(1.0 / b));
        public static final Initializer HE = (a, b) -> multiply(a, (float) Math.sqrt(2.0 / b));
    }

    /**
     * A set of common activation functions. The first parameter is a matrix of
     * values to be passed through a desired function. If the second parameter
     * is true, then the values are passed into the derivative of the desired
     * function.
     */
    public static class Activations {

        /**
         * A functional interface that is used for activation functions.
         */
        public interface Activation extends Serializable {

            /**
             * Applies the activation function to a matrix of values. The
             * derivative function determines if the derivative of the function
             * is to be applied to the values instead.
             *
             * @param matrix A 2d array with values to be passed into the
             * activation function.
             * @param derivative Indicates for the derivative of the function.
             * @return A 2d array with the function mapped to each value of the
             * given array.
             */
            float[][] apply(float[][] matrix, boolean derivative);
        }

        public static final Activation LINEAR = (matrix, derivative) -> {
            if (!derivative) {
                return matrix;
            } else {
                float[][] result = create(matrix.length, matrix[0].length, 1);
                return result;
            }
        };
        public static final Activation SIGMOID = (matrix, derivative) -> function(matrix, val -> sigmoid(val, derivative));
        public static final Activation TANH = (matrix, derivative) -> function(matrix, val -> tanh(val, derivative));
        public static final Activation RELU = (matrix, derivative) -> function(matrix, val -> relu(val, derivative));
        public static final Activation LEAKYRELU = (matrix, derivative) -> function(matrix, val -> leakyrelu(val, derivative));
        public static final Activation SWISH = (matrix, derivative) -> function(matrix, val -> swish(val, derivative));
        public static final Activation MISH = (matrix, derivative) -> function(matrix, val -> mish(val, derivative));
        public static final Activation SOFTMAX = (matrix, derivative) -> {
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
    public static class LossFunctions {

        public interface LossFunction extends Serializable {

            Object[] apply(float[][] outputs, float[][] targets);
        }

        /**
         * @param steepnessFactor Recommended value: .5
         * @return Quadratic loss function.
         */
        public static LossFunction QUADRATIC(double steepnessFactor) {
            float steepness = (float) steepnessFactor;
            return (float[][] outputs, float[][] targets) -> {
                double loss = sum(scale(steepness, square(subtract(outputs, targets))));//m(f(x) - y)^2 where f(x) is the output of the network and y is the target output
                return new Object[]{loss, scale(2 * steepness, subtract(outputs, targets))};//Derivative of the loss function for each sample, 2m(f(x) - y)
            };
        }

        /**
         * @param steepnessFactor Recommended value: 1
         * @return Huber loss function.
         */
        public static LossFunction HUBER(double steepnessFactor) {
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
        public static LossFunction HUBERPSEUDO(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                int columns = outputs[0].length;
                final float deltaSquared = steepness * steepness;
                final float[][] ones = create(1, columns, 1);
                final float[][] a = subtract(outputs, targets);
                final float[][] root = sqrt(add(ones, multiply(square(a), 1 / deltaSquared)));
                double loss = sum(scale(deltaSquared, subtract(root, ones)));
                return new Object[]{loss, divide(a, root)};
            };
        }

        /**
         * @param steepnessFactor Recommended value: 1
         * @return Cross entropy/log loss function.
         */
        public static LossFunction CROSSENTROPY(double steepnessFactor) {
            final float steepness = (float) steepnessFactor;
            return (outputs, targets) -> {
                double loss = steepness * -sum(multiply(targets, ln(outputs)));
                return new Object[]{loss, scale(-steepness, divide(targets, add(outputs, create(outputs.length, outputs[0].length, 1e-7f))))};//Preventing NaNs
            };
        }
    }

    /**
     * A set of common optimizers. The first three parameters of the
     * QuadFunction are strictly for the time step, learning rate, and the
     * gradients respectively. The third parameter is an array of matrices to
     * store info such as previous updates, gradients, etc. The QuadFunction
     * returns an array where its first element is a matrix with the gradient
     * update and its second element is the storage array(3 dimensional) that is
     * passed into the next call of the optimizer.
     */
    public static class Optimizers {

        public static final float beta = .9f;
        public static final float beta2 = .999f;
        public static final float e = 1e-7f;

        public interface Optimizer extends Serializable {

            Object[] apply(int step, float learningRate, float[][] gradients, float[][][] storage);
        }
        public static Optimizer VANILLA = (step, lr, gradients, storage) -> {
            float[][] update = scale(lr, gradients);
            return new Object[]{update, null};
        };
        public static Optimizer MOMENTUM = (step, lr, gradients, storage) -> {
            float[][] update = add(scale(beta, storage[0]), scale(lr, gradients));
            float[][][] store = {update};
            return new Object[]{update, store};
        };
        public static Optimizer NESTEROV = (step, lr, gradients, storage) -> {//Dozat's modification because calculating gradients of different parameters would take a lot of reworking
            float[][] m = add(scale(beta, storage[0]), gradients);
            float[][] update = scale(lr, add(gradients, scale(beta, m)));
            float[][][] store = {m};
            return new Object[]{update, store};
        };
        public static Optimizer ADAGRAD = (step, lr, gradients, storage) -> {
            float[][][] store = {add(storage[0], square(gradients))};
            float[][] update = multiply(scale(lr, sqrt(add(store[0], create(gradients.length, gradients[0].length, e)))), gradients);
            return new Object[]{update, store};
        };
        public static Optimizer ADADELTA = (step, lr, gradients, storage) -> {
            float[][] epsilon = create(gradients.length, gradients[0].length, e);
            float[][] gradientsE = ewma(beta, storage[0], square(gradients));
            float[][] gradientsRMS = sqrt(add(gradientsE, epsilon));
            float[][] deltaRMS = sqrt(add(storage[1], epsilon));
            float[][] update = multiply(divide(deltaRMS, gradientsRMS), gradients);
            float[][] deltaE = ewma(beta, storage[1], square(update));
            float[][][] store = {gradientsE, deltaE};
            return new Object[]{update, store};
        };
        public static Optimizer RMSPROP = (step, lr, gradients, storage) -> {
            float[][] s = ewma(beta, storage[0], square(gradients));
            float[][] s_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), s);
            float[][] update = divide(scale(lr, gradients), add(sqrt(s_), create(s.length, s[0].length, e)));
            float[][][] store = {s};
            return new Object[]{update, store};
        };
        public static Optimizer ADAM = (step, lr, gradients, storage) -> {
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] update = divide(scale(lr, m_), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
        public static Optimizer ADAMAX = (step, lr, gradients, storage) -> {
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = multiply(m, 1 / (1 - (float) Math.pow((double) beta, step)));
            float[][] u = max(scale(beta2, storage[1]), abs(gradients));
            float[][] update = divide(scale(lr, m_), add(u, create(u.length, u[0].length, e)));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
        public static Optimizer NADAM = (step, lr, gradients, storage) -> {
            int rows = gradients.length;
            int columns = gradients[0].length;
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] update = multiply(divide(create(rows, columns, lr), add(sqrt(v_), create(rows, columns, e))), add(scale(beta, m_), multiply(scale(1 - beta, gradients), 1 / (1 - beta))));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
        public static Optimizer AMSGRAD = (step, lr, gradients, storage) -> {
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] v__ = max(storage[1], v_);
            float[][] update = divide(scale(lr, m_), add(sqrt(v__), create(v__.length, v__[0].length, e)));
            float[][][] store = {m, v};
            return new Object[]{update, store};
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
            result[0][i] = (.5f * (tanh((.01f * ((oneRow[0][i] - mean) / (deviation))), false) + 1));
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
        if (!derivative) {
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
        float[][] result = functionMatrixVectors(matrix, vector -> subtract(vector, create(1, cols, max(vector))));//Stabilizing
        result = exp(result);
        return functionMatrixVectors(result, vector -> divide(vector, create(1, cols, sum(vector))));
    }

    private static class DotWorker extends Thread {

        int threadNum;
        int totalThreads;
        int rows1;
        int cols1;
        int cols2;
        float[][] m1;
        float[][] m2;
        float[][] result;

        DotWorker(int threadNum, int totalThreads, int rows1, int cols1, int cols2, float[][] m1, float[][] m2, float[][] result) {
            this.threadNum = threadNum;
            this.totalThreads = totalThreads;
            this.rows1 = rows1;
            this.cols1 = cols1;
            this.cols2 = cols2;
            this.m1 = m1;
            this.m2 = m2;
            this.result = result;
        }

        @Override
        public void run() {
            int endRow = (threadNum + 1) * rows1 / totalThreads;
            if (cols1 % 4 == 0) {//Loop unrolling increases speed
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k += 4) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j]
                                    + m1[i][k + 1] * m2[k + 1][j]
                                    + m1[i][k + 2] * m2[k + 2][j]
                                    + m1[i][k + 3] * m2[k + 3][j];
                        }
                    }
                }
            } else if (cols1 % 3 == 0) {
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k += 3) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j]
                                    + m1[i][k + 1] * m2[k + 1][j]
                                    + m1[i][k + 2] * m2[k + 2][j];
                        }
                    }
                }
            } else if (cols1 % 2 == 0) {
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k += 2) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j]
                                    + m1[i][k + 1] * m2[k + 1][j];
                        }
                    }
                }
            } else {
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k++) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j];
                        }
                    }
                }
            }
        }
    }

    public static float[][] dotThreads(float[][] m1, float[][] m2, int threads) {
        DotWorker[] threadArray = new DotWorker[threads];
        int rows1 = m1.length;
        int cols1 = m1[0].length;
        int cols2 = m2[0].length;
        float[][] result = new float[rows1][cols2];
        for (int t = 0; t < threads; t++) {
            threadArray[t] = new DotWorker(t, threads, rows1, cols1, cols2, m1, m2, result);
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

    public static float[][] dot(float[][] m1, float[][] m2) {
        int rows1 = m1.length;
        int cols1 = m1[0].length;
        int cols2 = m2[0].length;
        float[][] result = new float[rows1][cols2];
        if (cols1 % 4 == 0) {//Loop unrolling increases speed
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k += 4) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j]
                                + m1[i][k + 2] * m2[k + 2][j]
                                + m1[i][k + 3] * m2[k + 3][j];
                    }
                }
            }
        } else if (cols1 % 3 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k += 3) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j]
                                + m1[i][k + 2] * m2[k + 2][j];
                    }
                }
            }
        } else if (cols1 % 2 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k += 2) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j];
                    }
                }
            }
        } else {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k++) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
        }
        return result;
    }

    public static float[][] dotVector(float[] vector, float[][] matrix) {
        int rows2 = matrix.length;
        int cols2 = matrix[0].length;
        float[][] result = new float[1][cols2];
        if (rows2 % 4 == 0) {
            for (int i = 0; i < rows2; i += 4) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j]
                            + vector[i + 1] * matrix[i + 1][j]
                            + vector[i + 2] * matrix[i + 2][j]
                            + vector[i + 3] * matrix[i + 3][j];
                }
            }
        } else if (rows2 % 3 == 0) {
            for (int i = 0; i < rows2; i += 3) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j]
                            + vector[i + 1] * matrix[i + 1][j]
                            + vector[i + 2] * matrix[i + 2][j];
                }
            }
        } else if (rows2 % 2 == 0) {
            for (int i = 0; i < rows2; i += 2) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j]
                            + vector[i + 1] * matrix[i + 1][j];
                }
            }
        } else {
            for (int i = 0; i < rows2; i++) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j];
                }
            }
        }
        return result;
    }

    public static String arrToString(float[][] arr2d) {
        int rowLastIndex = arr2d.length - 1;
        int columns = arr2d[0].length;
        String string = "[";
        for (int i = 0; i < rowLastIndex; i++) {
            string += "[" + arr2d[i][0];
            for (int j = 1; j < columns; j++) {
                string += " " + arr2d[i][j];
            }
            string += "]\n";
        }
        string += "[" + arr2d[rowLastIndex][0];
        for (int j = 1; j < columns; j++) {
            string += " " + arr2d[rowLastIndex][j];
        }
        string += "]]\n";
        return string;
    }

    public static String arrToString(float[][][] arr3d) {
        int depth = arr3d.length;
        String string = "[\n" + arrToString(arr3d[0]);
        for (int i = 1; i < depth; i++) {
            string += "\n" + arrToString(arr3d[i]);
        }
        return string + "]\n";
    }

    public static void print(float[][] arr2d) {
        System.out.print(arrToString(arr2d));
    }

    public static void print(float[][] arr2d, String label) {
        System.out.println(label + ":");
        print(arr2d);
    }

    public static void print(float[][][] arr3d) {
        int depth = arr3d.length;
        System.out.println("[");
        System.out.print(arrToString(arr3d[0]));
        for (int i = 1; i < depth; i++) {
            System.out.print("\n" + arrToString(arr3d[i]));
        }
        System.out.println("]");
    }

    public static void print(float[][][] arr3d, String label) {
        System.out.println(label + ":");
        print(arr3d);
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

    public static float[] create(int columns, float valueToAllElements) {
        float[] result = new float[columns];
        for (int i = 0; i < columns; i++) {
            result[i] = valueToAllElements;
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

    public static float[][][] create(int depth, int rows, int columns, float valueToAllElements) {
        float[][][] result = new float[depth][][];
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < columns; k++) {
                    result[i][j][k] = valueToAllElements;
                }
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

    public static float[][] multiply(float[][] matrix, float factor) {
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

    public static float[][] ewma(float beta, float[][] prevStep, float[][] currentStep) {
        return add(scale(beta, prevStep), scale(1 - beta, currentStep));
    }

    /**
     * Replaces the maximum value in the given matrix with a 1 and replace the
     * rest of the elements with 0s.
     *
     * @param matrix a matrix with a unique maximum, otherwise the first maximum
     * value is chosen.
     * @return The resulting matrix.
     */
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

    /**
     * Creates a new matrix with the given dimensions and with a one in the
     * given position and 0s in the rest.
     *
     * @param rows Height of the result.
     * @param cols Width of the result.
     * @param onehotRow The row of the 1 in the result.
     * @param onehotCol The column of the 1 in the result.
     * @return The resulting matrix.
     */
    public static float[][] oneHot(int rows, int cols, int onehotRow, int onehotCol) {
        float[][] result = new float[rows][cols];
        result[onehotRow][onehotCol] = 1;
        return result;
    }

    /**
     * Samples from the given probabilities. Works properly when probabilities
     * add up to 1
     *
     * @param probabilities Array of probabilities adding up to 1.
     * @return The chosen index. Depending on the probabilities, will return -1
     * if probabilities don't add up to 1 and it resulted in no determined
     * index.
     */
    public static int sampleProbabilities(float[] probabilities) {
        float random = (float) Math.random();
        int size = probabilities.length;
        for (int i = 0; i < size; i++) {
            random -= probabilities[i];
            if (random < 0) {
                return i;
            }
        }
        return -1;
    }

    /**
     * @param probabilities Array of probabilities adding up to 1.
     * @param rng Random class for seeded randomness
     * @return The chosen index. Depending on the probabilities, will return -1
     * if probabilities don't add up to 1 and it resulted in no determined
     * index.
     * @see #sampleProbabilities(float[])
     */
    public static int sampleProbabilities(float[] probabilities, Random rng) {
        float random = rng.nextFloat();
        int size = probabilities.length;
        for (int i = 0; i < size; i++) {
            random -= probabilities[i];
            if (random < 0) {
                return i;
            }
        }
        return -1;
    }

    public static int[] includeElements(int[] array, int... indeces) {
        int length = indeces.length;
        int[] newArray = new int[length];
        for (int i = 0; i < length; i++) {
            newArray[i] = array[indeces[i]];
        }
        return newArray;
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

    public static int getDimensionality(Object[] arr) {
        int dimensions = 0;
        Class type = arr.getClass();
        while (type.isArray()) {
            dimensions++;
            type = type.getComponentType();
        }
        return dimensions;
    }

    public static String getDimensions(Object[] arr) {
        String dimensions = arr.length + " ";
        try {
            while (arr.getClass().getComponentType().isArray()) {
                try {
                    arr = (Object[]) arr[0];
                    dimensions += arr.length + " ";
                } catch (ClassCastException f) {
                    try {
                        dimensions += ((float[]) (arr[0])).length;
                    } catch (ClassCastException d) {
                        dimensions += ((double[]) (arr[0])).length;
                    }
                    break;
                } catch (NullPointerException n) {
                    dimensions += "null ";
                }
            }
        } catch (NullPointerException n) {
            dimensions += "null ";
        }
        return dimensions.trim();
    }

    public static Class getDeepType(Object[] arr) {
        Class type = arr.getClass().getComponentType();
        while (type.isArray()) {
            type = type.getComponentType();
        }
        return type;
    }

    public static float[] copy(float[] arr1d) {
        int length = arr1d.length;
        float[] result = new float[length];
        System.arraycopy(arr1d, 0, result, 0, length);
        return result;
    }

    public static float[][] copy(float[][] arr2d) {
        return Arrays.stream(arr2d).map(el -> el.clone()).toArray(a -> arr2d.clone());
    }

    public static float[][][] copy(float[][][] arr3d) {
        return Arrays.stream(arr3d).map(el -> copy(el)).toArray(a -> arr3d.clone());
    }

    public static float[][][][] copy(float[][][][] arr4d) {
        return Arrays.stream(arr4d).map(el -> copy(el)).toArray(a -> arr4d.clone());
    }

    public static float[][][][][] copy(float[][][][][] arr5d) {
        return Arrays.stream(arr5d).map(el -> copy(el)).toArray(a -> arr5d.clone());
    }

    public static Object[] copy(Object[] nDimensionArr) {
        try {
            return Arrays.stream(nDimensionArr).map(el -> copy(Object[].class.cast(el))).toArray(a -> nDimensionArr.clone());
        } catch (ClassCastException e) {
            return Arrays.stream(nDimensionArr).map(el -> copy(float[].class.cast(el))).toArray(a -> nDimensionArr.clone());
        }
    }

    public static float[] flatten(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[(i * cols) + j] = matrix[i][j];
            }
        }
        return result;
    }

    public static float[][] unflatten(float[] flattened, int rows, int cols) {
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = flattened[(i * cols) + j];
            }
        }
        return result;
    }

    public static float[][] convolution(float[][] input, float[][] filter, float bias, int stride) {
        int filterHeight = filter.length;
        int filterWidth = filter[0].length;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int resultHeight = (inputHeight - filterHeight) / stride + 1;
        int resultWidth = (inputWidth - filterWidth) / stride + 1;
        float[][] result = new float[resultHeight][resultWidth];
        for (int rRow = 0; rRow < resultHeight; rRow++) {
            int startRow = rRow * stride;
            int boundRow = startRow + filterHeight;
            for (int rCol = 0; rCol < resultWidth; rCol++) {
                int startCol = rCol * stride;
                int boundCol = startCol + filterWidth;
                float val = 0;
                for (int iRow = startRow; iRow < boundRow; iRow++) {
                    int filterRow = iRow - startRow;
                    for (int iCol = startCol; iCol < boundCol; iCol++) {
                        val += input[iRow][iCol] * filter[filterRow][iCol - startCol];
                    }
                }
                result[rRow][rCol] = val + bias;
            }
        }
        return result;
    }

    public static float[][][] convolution(float[][][] input, float[][][][] filters, float[] biases, int stride) {
        int resultDepth = filters.length;
        int filterDepth = filters[0].length;
        int filterHeight = filters[0][0].length;
        int filterWidth = filters[0][0][0].length;
        int resultHeight = (input[0].length - filterHeight) / stride + 1;
        int resultWidth = (input[0][0].length - filterWidth) / stride + 1;
        float[][][] result = new float[resultDepth][resultHeight][resultWidth];
        for (int rDepth = 0; rDepth < resultDepth; rDepth++) {
            float[][][] filter = filters[rDepth];
            for (int rRow = 0; rRow < resultHeight; rRow++) {
                int startRow = rRow * stride;
                int boundRow = startRow + filterHeight;
                for (int rCol = 0; rCol < resultWidth; rCol++) {
                    int startCol = rCol * stride;
                    int boundCol = startCol + filterWidth;
                    float val = 0;
                    for (int iDepth = 0; iDepth < filterDepth; iDepth++) {
                        float[][] filterLayer = filter[iDepth];
                        float[][] inputLayer = input[iDepth];
                        for (int iRow = startRow; iRow < boundRow; iRow++) {
                            int filterRow = iRow - startRow;
                            for (int iCol = startCol; iCol < boundCol; iCol++) {
                                val += inputLayer[iRow][iCol] * filterLayer[filterRow][iCol - startCol];
                            }
                        }
                    }
                    result[rDepth][rRow][rCol] = val + biases[rDepth];
                }
            }
        }
        return result;
    }

    public static float[][] rotate180(float[][] arr2d) {
        int rows = arr2d.length;
        int cols = arr2d[0].length;
        float[][] result = new float[rows][cols];
        int maxrow = rows - 1;
        int maxcol = cols - 1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[maxrow - i][maxcol - j] = arr2d[i][j];
            }
        }
        return result;
    }

    public static float[][] pad(float[][] arr2d, int paddingHeight, int paddingWidth) {
        if (!(paddingHeight < 0 || paddingWidth < 0) || (paddingHeight == 0 && paddingWidth == 0)) {
            int paddedHeight = arr2d.length + paddingHeight + paddingHeight;
            int paddedWidth = arr2d[0].length + paddingWidth + paddingWidth;
            float[][] padded = new float[paddedHeight][paddedWidth];
            int endRow = paddedHeight - paddingHeight;
            int endCol = paddedWidth - paddingWidth;
            for (int j = 0; j < paddingHeight; j++) {//Top
                for (int k = 0; k < paddingWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = endRow; j < paddedHeight; j++) {//Bottom
                for (int k = endCol; k < paddedWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = paddingHeight; j < paddedHeight; j++) {//Left
                for (int k = 0; k < paddingWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = paddingHeight; j < paddedHeight; j++) {//Right
                for (int k = endCol - 1; k < paddingWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = paddingHeight; j < endRow; j++) {//Matrix numbers
                for (int k = paddingWidth; k < endCol; k++) {
                    padded[j][k] = arr2d[j - paddingHeight][k - paddingWidth];
                }
            }
            return padded;
        } else {
            return arr2d;
        }
    }

    public static float[][] unpad(float[][] arr2d, int unpaddingHeight, int unpaddingWidth) {
        if (!(unpaddingHeight < 0 || unpaddingWidth < 0) || (unpaddingHeight == 0 || unpaddingWidth == 0)) {
            int resultHeight = arr2d.length - unpaddingHeight - unpaddingHeight;
            int resultWidth = arr2d[0].length - unpaddingWidth - unpaddingWidth;
            float[][] result = new float[resultHeight][resultWidth];
            for (int i = 0; i < resultHeight; i++) {
                for (int j = 0; j < resultWidth; j++) {
                    result[i][j] = arr2d[i + unpaddingHeight][j + unpaddingWidth];
                }
            }
            return result;
        } else {
            return arr2d;
        }
    }

    public static float[][] dilate(float[][] arr2d, int dilationRows, int dilationCols) {
        if (!(dilationRows < 0 || dilationCols < 0) || (dilationRows == 0 || dilationCols == 0)) {
            int rows = arr2d.length;
            int cols = arr2d[0].length;
            float[][] result = new float[rows * (dilationRows + 1) - dilationRows][cols * (dilationCols + 1) - dilationCols];
            int jumpCols = dilationCols + 1;
            int jumpRows = dilationRows + 1;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i * jumpRows][j * jumpCols] = arr2d[i][j];
                }
            }
            return result;
        } else {
            return arr2d;
        }
    }

    public static float[][][] function2dOn3d(float[][][] arr3d, Function<float[][], float[][]> function) {
        int depth = arr3d.length;
        float[][][] result = new float[depth][][];
        for (int i = 0; i < depth; i++) {
            result[i] = function.apply(arr3d[i]);
        }
        return result;
    }

    public static float[][][] bifunction2dOn3d(float[][][] arr3d1, float[][][] arr3d2, BiFunction<float[][], float[][], float[][]> bifunction) {
        int depth = arr3d1.length;
        float[][][] result = new float[depth][][];
        for (int i = 0; i < depth; i++) {
            result[i] = bifunction.apply(arr3d1[i], arr3d2[i]);
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
            xAxis.setLabel("Steps");
            xAxis.setForceZeroInRange(false);
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
                    series.getData().add(new XYChart.Data<>(nn.iterations, 1 / Math.pow(100 * Math.E, nn.loss)));
                } else {
                    series.getData().add(new XYChart.Data<>(nn.iterations, nn.loss));
                }
                if (series.getData().size() > 30000) {
                    series.getData().remove(0);
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
                    launch(NNlib.class);
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
