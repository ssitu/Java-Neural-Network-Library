package neuralnetwork;

import java.util.Random;
import neuralnetwork.NNLib.*;

public class XOR_Classification {

    public static void main(String[] args) {
        final boolean PRINT = true;
        final BiFunction<float[][], Boolean, float[][]> CUSTOM = (matrix, derivative) -> {//Custom activation function -> max(tanh(x),x)
            int rows = matrix.length;
            int columns = matrix[0].length;
            if (!derivative) {
                return NNLib.max(NNLib.function(matrix, val -> NNLib.tanh(val, false)), matrix);
            } else {
                float[][] result = new float[rows][columns];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < columns; j++) {
                        float val = matrix[i][j];
                        if (NNLib.tanh(val, false) > val) {
                            result[i][j] = NNLib.tanh(val, true);
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
                .001f,//Learning Rate for Optimizer
                LossFunction.CROSSENTROPY(1),//Loss/Cost/Error Function
                Optimizer.NESTEROV,//Gradient Descent Optimizer
                new Layer.Dense(2, 3, CUSTOM, Initializer.XAVIER),
                new Layer.Dense(3, 2, ActivationFunction.SOFTMAX, Initializer.XAVIER)
        );
        System.out.println("Seed: " + seed);
        System.out.println("NN Length: " + nn.length);
        System.out.println(nn);
        float[][][] data = {
            {//Inputs
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
            },
            {//Labels: first output node = true, second output node = false;
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
            }
        };
        NNLib.showInfo(NNLib.infoLayers, nn);
        NNLib.showInfo(NNLib.infoGraph(false), nn);
        for (int i = 0; i < 100_000_000; i++) {
            int index = nn.getRandom().nextInt(4);
            nn.backpropagation(new float[][]{data[0][index]}, new float[][]{data[1][index]});
//            nn.backpropagation(data[0], data[1]);
            if (PRINT && i % 100_000 == 0) {
                NNLib.print(data[0], "Dataset Inputs");
                NNLib.print(nn.feedforward(data[0]), "Outputs");
                System.out.println("");
            }
        }
        System.exit(0);
    }
}
