package neuralnetwork;

import java.util.ArrayList;
import neuralnetwork.NNLib.*;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        NN nn = new NNLib().new NN(
                "xor",//Name for Saving & Graph Title
                7777,//Seed For Reproducibility
                .1,//Learning Rate for Optimizer
                Initializer.VANILLA,//Weight & Bias Initializer Method
                ActivationFunction.SIGMOID,//Hiddens
                ActivationFunction.SIGMOID,//Outputs
                LossFunction.QUADRATIC.steepness(2),//Loss/Cost/Error Function
                Optimizer.VANILLA,//Stochastic Gradient Descent Optimizer
                2, 2, 1//Network Architecture
        );
        System.out.println(nn.NETWORKSIZE);
        System.out.println(nn.toString());
        ArrayList<Data> data = new ArrayList<>();

        //XOR truth table
        data.add(new Data(new float[]{1, 1}, new float[]{0}));//True, True = False
        data.add(new Data(new float[]{0, 1}, new float[]{1}));//False, True = True
        data.add(new Data(new float[]{1, 0}, new float[]{1}));//True, False = True
        data.add(new Data(new float[]{0, 0}, new float[]{0}));//False, False = False
        NNLib.graph(false, nn);
        for (int i = 0; i < 10_000_000; i++) {
            int index = nn.getRandom().nextInt(4);
            if (PRINT) {
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
