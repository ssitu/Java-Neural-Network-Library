package nnlibrary.hyperparameters;

import java.io.Serializable;
import static nnlibrary.hyperparameters.Functions.add;
import static nnlibrary.hyperparameters.Functions.create;
import static nnlibrary.hyperparameters.Functions.divide;
import static nnlibrary.hyperparameters.Functions.ln;
import static nnlibrary.hyperparameters.Functions.multiply;
import static nnlibrary.hyperparameters.Functions.scale;
import static nnlibrary.hyperparameters.Functions.sqrt;
import static nnlibrary.hyperparameters.Functions.square;
import static nnlibrary.hyperparameters.Functions.subtract;
import static nnlibrary.hyperparameters.Functions.sum;

/**
     * A set of common loss functions. First parameter takes in a matrix of
     * outputs from the last layer of the network. The seconds parameter is a
     * matrix of targets with the same shape of the outputs. The return value is
     * an Object array where the first index is the result of the loss function
     * given the outputs and targets and the second index is a matrix with the
     * same shape as the outputs and targets with values from the derivative of
     * the loss function with the outputs as inputs.
     */
    public class LossFunctions {

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