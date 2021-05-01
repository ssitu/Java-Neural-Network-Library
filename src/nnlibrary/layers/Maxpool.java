package nnlibrary.layers;

import java.io.Serializable;
import java.util.Random;
import static nnlibrary.hyperparameters.Functions.getDimensions;
import nnlibrary.hyperparameters.Optimizers;

public class Maxpool extends Layer implements Serializable {

            private static final long serialVersionUID = 0;
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
             * @see Layer#initialize(nnlibrary.Functions.Layer,
             * java.util.Random)
             */
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
             * nnlibrary.Functions.Optimizers.Optimizer, int, boolean)
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