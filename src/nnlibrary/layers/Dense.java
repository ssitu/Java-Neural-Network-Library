package nnlibrary.layers;

import java.io.Serializable;
import java.util.Random;
import nnlibrary.Init;
import static nnlibrary.Init.dotProduct;
import nnlibrary.hyperparameters.Activations;
import nnlibrary.hyperparameters.Functions;
import static nnlibrary.hyperparameters.Functions.add;
import static nnlibrary.hyperparameters.Functions.append;
import static nnlibrary.hyperparameters.Functions.arrToString;
import static nnlibrary.hyperparameters.Functions.copy;
import static nnlibrary.hyperparameters.Functions.multiply;
import static nnlibrary.hyperparameters.Functions.subtract;
import static nnlibrary.hyperparameters.Functions.transpose;
import nnlibrary.hyperparameters.Initializers;
import nnlibrary.hyperparameters.Optimizers;

public class Dense extends Layer implements Serializable {

            private static final long serialVersionUID = Init.serial;
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
                weights = Functions.randomize(new float[nodesIn][nodesOut], 2, -1, random);//values on interval [-1,1]
                biases = Functions.randomize(new float[1][nodesOut], 2, -1, random);//values on interval [-1,1]
                weights = initializer.apply(weights, nodesIn);
                updateStorageW = new float[1][nodesIn][nodesOut];
                updateStorageB = new float[1][1][nodesOut];
                accumulatedW = new float[nodesIn][nodesOut];
                accumulatedB = new float[1][nodesOut];
                super.OUTSHAPE = new int[]{nodesOut};
            }

            /**
             * @see Layer#initialize(nnlibrary.Functions.Layer,
             * java.util.Random)
             */
            @Override
            public void initialize(Layer previousLayer, Random random) {
                nodesIn = previousLayer.OUTSHAPE[0];
                weights = Functions.randomize(new float[nodesIn][nodesOut], 2, -1, random);//values on interval [-1,1]
                biases = Functions.randomize(new float[1][nodesOut], 2, -1, random);//values on interval [-1,1]
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
             * nnlibrary.Functions.Optimizers.Optimizer, int, boolean)
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
                weights = Functions.randomize(weights, range, -range / 2);
                biases = Functions.randomize(biases, range, -range / 2);
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
