package nnlibrary.layers;

import java.io.Serializable;
import java.util.Random;
import nnlibrary.hyperparameters.Activations;
import nnlibrary.hyperparameters.Initializers;
import nnlibrary.hyperparameters.Optimizers;

/**
 * A set of layer types.
 */
public abstract class Layer implements Serializable {

    public final boolean INFER;
    protected int[] OUTSHAPE;

    public Layer(boolean infer) {
        INFER = infer;
    }

    /**
     * Initializes this Layer's parameters and storage of previous gradients.
     *
     * @param random A Random instance to create random values from.
     */
    public abstract void initialize(Random random);

    /**
     * Initializes this Layer's parameters and storage of previous gradients.
     * With inference of input shape from the prior layer.
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
     * @param dC_dA_uncasted The partial derivatives of the loss with respect to
     * the activations of the current Layer.
     * @param lr The learning rate for the optimizer.
     * @param optimizer The optimizer to be used to tune the parameters.
     * @param step The number of times the parameters have been tuned. This can
     * be set to 0 but will cause biased gradients.
     * @param update Whether or not to tune the parameters after computing
     * gradients.
     * @return The partial derivatives of the loss with respect to the values
     * before the activation function.
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
     * Summarizes the details of this Layer into a String. Needed in the save
     * method in the NN class, so it must leave out characters that cannot be
     * used in a file name.
     *
     * @return A String with the important specifications of this Layer, leaving
     * out the parameters.
     */
    @Override
    public abstract String toString();
}
