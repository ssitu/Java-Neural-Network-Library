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
import static nnlibrary.hyperparameters.Functions.bifunction2dOn3d;
import static nnlibrary.hyperparameters.Functions.convolution;
import static nnlibrary.hyperparameters.Functions.copy;
import static nnlibrary.hyperparameters.Functions.dilate;
import static nnlibrary.hyperparameters.Functions.function2dOn3d;
import static nnlibrary.hyperparameters.Functions.getDimensions;
import static nnlibrary.hyperparameters.Functions.multiply;
import static nnlibrary.hyperparameters.Functions.pad;
import static nnlibrary.hyperparameters.Functions.rotate180;
import static nnlibrary.hyperparameters.Functions.subtract;
import static nnlibrary.hyperparameters.Functions.sum;
import nnlibrary.hyperparameters.Optimizers;

public class Conv extends Layer implements Serializable {

            private static final long serialVersionUID = Init.serial;
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
            private final Functions.Function<float[][], float[][]> pad;
            private final Functions.Function<float[][], float[][]> unpad;
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
                pad = a -> Functions.pad(a, paddingHeight, paddingWidth);
                unpad = a -> Functions.unpad(a, paddingHeight, paddingWidth);
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
                pad = a -> Functions.pad(a, paddingHeight, paddingWidth);
                unpad = a -> Functions.unpad(a, paddingHeight, paddingWidth);
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
                        filters[i][j] = Functions.randomize(new float[filterHeight][filterWidth], 2, -1, random);
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
             * @see Layer#initialize(nnlibrary.Functions.Layer,
             * java.util.Random)
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
                        filters[i][j] = Functions.randomize(new float[filterHeight][filterWidth], 2, -1, random);
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
                    Z = Functions.convolution(prevA, filters, biases, stride);
                    return function2dOn3d(Z, a -> activation.apply(a, false));
                } catch (ArrayIndexOutOfBoundsException e) {
                    throw new IllegalArgumentException("Input dimensions are " + getDimensions(in) + " but the filter size is " + getDimensions(filters));
                }
            }

            /**
             * @see Layer#back(java.lang.Object[], float,
             * nnlibrary.Functions.Optimizers.Optimizer, int, boolean)
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
                        Functions.randomize(filters[i][j], range, -range / 2);
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