package neuralnetwork;

import java.util.ArrayList;
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
                .001,//Learning Rate for Optimizer
                LossFunction.CROSSENTROPY(1),//Loss/Cost/Error Function
                Optimizer.NESTEROV,//Stochastic Gradient Descent Optimizer
                new Layer.Dense(2, 3, CUSTOM, Initializer.XAVIER),
                new Layer.Dense(3, 2, ActivationFunction.SOFTMAX, Initializer.XAVIER)
        );
        System.out.println("Seed: " + seed);
        System.out.println(nn.length);
        System.out.println(nn.toString());
        ArrayList<Data> data = new ArrayList<>();

        //First output = true, second output = false;
        data.add(new Data(new float[]{1, 1}, new float[]{0, 1}));
        data.add(new Data(new float[]{0, 1}, new float[]{1, 0}));
        data.add(new Data(new float[]{1, 0}, new float[]{1, 0}));
        data.add(new Data(new float[]{0, 0}, new float[]{0, 1}));
        NNLib.showInfo(NNLib.infoLayers, nn);
        NNLib.showInfo(NNLib.infoGraph(false), nn);
        for (int i = 0; i < 100_000_000; i++) {
            int index = nn.getRandom().nextInt(4);
            if (PRINT && i % 100_000 == 0) {
                System.out.println("Inputs:");
                NNLib.print(data.get(index).inputs);
                System.out.println("Outputs:");
                NNLib.print(nn.feedforward(data.get(index).inputs));
                System.out.println("");
            }
            nn.backpropagation(data.get(index).inputs, data.get(index).targets);
        }
        System.exit(0);
    }

    static class Data {

        float[][] inputs;
        float[][] targets;

        Data(float[] inputs, float[] targets) {
            this.inputs = new float[][]{inputs};
            this.targets = new float[][]{targets};
        }
    }
}
