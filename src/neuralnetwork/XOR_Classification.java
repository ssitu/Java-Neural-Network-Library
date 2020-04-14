package neuralnetwork;

import java.util.ArrayList;
import java.util.function.BiFunction;
import neuralnetwork.NNLib.*;

public class XOR_Classification {

    public static void main(String[] args) {
        final boolean PRINT = true;
        final BiFunction<float[][], Boolean, float[][]> CUSTOM = (matrix, derivative) -> {//Custom activation function
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
        NN nn = new NNLib().new NN(
                "xor_classification",//Name for Saving & Graph Title
                7777,//Seed For Reproducibility
                .001,//Learning Rate for Optimizer
                LossFunction.CROSSENTROPY(),//Loss/Cost/Error Function
                Optimizer.AMSGRAD,//Stochastic Gradient Descent Optimizer
                new LayerDense(2, 2, CUSTOM, Initializer.HE),
                new LayerDense(2, 1, ActivationFunction.SIGMOID, Initializer.XAVIER)
        );
        System.out.println(nn.NETWORKSIZE);
        System.out.println(nn.toString());
        ArrayList<Data> data = new ArrayList<>();

        //First output = true, second output = false;
        data.add(new Data(new float[]{1, 1}, new float[]{0, 1}));
        data.add(new Data(new float[]{0, 1}, new float[]{1, 0}));
        data.add(new Data(new float[]{1, 0}, new float[]{1, 0}));
        data.add(new Data(new float[]{0, 0}, new float[]{0, 1}));
        NNLib.graph(false, nn);
        for (int i = 0; i < 1_000_000; i++) {
            int index = nn.getRandom().nextInt(4);
            if (PRINT && i % 1000 == 0) {
                NNLib.print(data.get(index).inputs, "inputs");
                NNLib.print(nn.feedforward(data.get(index).inputs), "feedforward");
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
