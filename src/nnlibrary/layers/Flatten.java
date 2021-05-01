package nnlibrary.layers;

import java.io.Serializable;
import java.util.Random;
import nnlibrary.Init;
import nnlibrary.hyperparameters.Optimizers;

public class Flatten extends Layer implements Serializable {

    private static final long serialVersionUID = Init.serial;
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
     * @see Layer#initialize(nnlibrary.Functions.Layer, java.util.Random)
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
     * nnlibrary.Functions.Optimizers.Optimizer, int, boolean)
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
