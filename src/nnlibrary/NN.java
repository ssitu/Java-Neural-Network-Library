package nnlibrary;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;
import nnlibrary.hyperparameters.LossFunctions;
import nnlibrary.hyperparameters.Optimizers;
import nnlibrary.layers.Layer;

/**
 * The neural network class. This manages all the layers that are put inside the
 * constructor for easy operations. Having a NN instance is optional for a
 * neural network since all the operations can be done with the Layer instances.
 *
 */
public class NN implements Serializable {

    private static final long serialVersionUID = Init.serial;
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
     * The name of this NN instance. Used in saving, loading, and info panels.
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
     * Some variable names do not follow convention so that when the IDE lists
     * out the variables when creating an NN instance you would only have to add
     * a period followed by the name of the already implemented function e.g.
     * Optimizers.ADAM.
     *
     * @param label The name for the NN.
     * @param seed A seed for repeatable Layer initialization.
     * @param learningRate A value 0 to 1 for training layer parameters.
     * @param LossFunctions Measures the error between two one row matrices.
     * @param Optimizers An algorithm that speeds up SGD, or use the VANILLA
     * optimizer for regular SGD.
     * @param Layer The Layer in which inputs first feeds into. Cannot infer
     * input shape since it is the first Layer in the network so it must not be
     * created with the constructor that leaves out the input shape.
     * @param Layers A list of Layers following the input Layer to be used in
     * the network.
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
     * @param targets The desired output for the network to fit to. Must match
     * the shape of the outputs or either an ArrayOutOfBoundsException will be
     * thrown or some of the targets will be ignored.
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
     * Copies the parameters from a NN to this NN instance, assuming the network
     * architectures and hyperparameters of both NNs are identical for this to
     * work properly.
     *
     * @param nn The neural network to copy from.
     */
    public void copyParameters(NN nn) {
        for (int i = 0; i < nn.length; i++) {
            network[i] = nn.network[i].clone();
        }
    }

    /**
     * Get the total number of adjustable(through backpropagation) parameters in
     * this NN.
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
     * Loads a serialized NN instance created by {@link #save()}. Will search
     * for the file with the same NN label and layer architecture in the name.
     * Careful loading after changing the NN hyper parameters. The directory is
     * in the current user folder if used as a jar executable
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
     * Same as {@link #load()} method but instead will look outside of the .jar.
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
     * Same as {@link #load()} method but instead will look inside of the .jar.
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
     * inputs. Passing in one less than the length of the network would return
     * the output layer.
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
     * Get the total amount of times the network has done backpropagation. This
     * number resets everytime the optimizer is changed.
     *
     * @return Times the backpropagation method has been called with the current
     * optimizer.
     */
    public int getSteps() {
        return step;
    }

    /**
     * Get the recent loss value from the call of backpropagation.
     *
     * @return The most recent measured loss from backpropagation.
     */
    public double getLoss() {
        return loss;
    }

    public long getIterations() {
        return iterations;
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
     * @param learningRate A double that is casted to a float to avoid having to
     * work with floats.
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
