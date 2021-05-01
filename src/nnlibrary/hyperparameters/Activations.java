package nnlibrary.hyperparameters;

import java.io.Serializable;
import static nnlibrary.hyperparameters.Functions.create;
import static nnlibrary.hyperparameters.Functions.function;
import static nnlibrary.hyperparameters.Functions.leakyrelu;
import static nnlibrary.hyperparameters.Functions.mish;
import static nnlibrary.hyperparameters.Functions.relu;
import static nnlibrary.hyperparameters.Functions.sigmoid;
import static nnlibrary.hyperparameters.Functions.softmax;
import static nnlibrary.hyperparameters.Functions.swish;
import static nnlibrary.hyperparameters.Functions.tanh;

/**
     * A set of common activation functions. The first parameter is a matrix of
     * values to be passed through a desired function. If the second parameter
     * is true, then the values are passed into the derivative of the desired
     * function.
     */
    public class Activations {

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
