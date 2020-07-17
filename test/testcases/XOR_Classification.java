package testcases;

import nnlibrary.NNlib;
import nnlibrary.NNlib.*;
import java.util.Random;

public class XOR_Classification {

    public static void main(String[] args) {
        final boolean PRINT = true;
        final BiFunction<float[][], Boolean, float[][]> CUSTOM = (matrix, derivative) -> {//Custom activation function -> max(tanh(x),x)
            int rows = matrix.length;
            int columns = matrix[0].length;
            if (!derivative) {
                return NNlib.max(NNlib.function(matrix, val -> NNlib.tanh(val, false)), matrix);
            } else {
                float[][] result = new float[rows][columns];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < columns; j++) {
                        float val = matrix[i][j];
                        if (NNlib.tanh(val, false) > val) {
                            result[i][j] = NNlib.tanh(val, true);
                        } else {
                            result[i][j] = 1;
                        }
                    }
                }
                return result;
            }
        };
        long seed = new Random().nextLong();
        NN nn = new NN(
                "xor_classification",//Name for Saving & Graph Title
                seed,//Seed For Reproducibility
                0,//Learning Rate for Optimizer
                LossFunction.CROSSENTROPY(1),//Loss/Cost/Error Function
                Optimizer.ADADELTA,//Gradient Descent Optimizer
                new Layer.Dense(2, 4, CUSTOM, Initializer.XAVIER),
                new Layer.Dense(4, 2, Activation.SOFTMAX, Initializer.XAVIER)
        );
        nn.setBatchSize(4);
        System.out.println("Seed: " + seed);
        System.out.println("NN Length: " + nn.length);
        System.out.println(nn);
        float[][][][] data = {
            {//Inputs
                {{0, 0}},
                {{0, 1}},
                {{1, 0}},
                {{1, 1}}
            },
            {//Labels: first output node = true, second output node = false;
                {{0, 1}},
                {{1, 0}},
                {{1, 0}},
                {{0, 1}}
            }
        };
        NNlib.showInfo(NNlib.infoLayers, nn);
        NNlib.showInfo(NNlib.infoGraph(false), nn);
        for (int i = 0; i < 100_000_000; i++) {
            if (i % 4 == 0) {
                nn.backpropagation(data[0][0], data[1][0]);
            } else if (i % 4 == 1) {
                nn.backpropagation(data[0][1], data[1][1]);
            } else if (i % 4 == 2) {
                nn.backpropagation(data[0][2], data[1][2]);
            } else if (i % 4 == 3) {
                nn.backpropagation(data[0][3], data[1][3]);
            }
            if (PRINT && i % 100000 == 0) {
                System.out.println("XOR Inputs:");
                float[][] inputs = NNlib.append(data[0][0], data[0][1], data[0][2], data[0][3]);
                NNlib.print(inputs);
                System.out.println("Outputs:");
                NNlib.print((float[][]) nn.feedforward(data[0][0]));
                NNlib.print((float[][]) nn.feedforward(data[0][1]));
                NNlib.print((float[][]) nn.feedforward(data[0][2]));
                NNlib.print((float[][]) nn.feedforward(data[0][3]));
                System.out.println("");
            }
        }
        System.exit(0);
    }
}
