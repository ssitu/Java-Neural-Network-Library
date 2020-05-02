package neuralnetwork;

import java.util.ArrayList;
import neuralnetwork.NNLib.*;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        NN nn = new NNLib().new NN(
                "xor",//Name for Saving & Graph Title
                777,//Seed For Reproducibility
                .001,//Learning Rate for Optimizer
                LossFunction.QUADRATIC(.5),//Loss/Cost/Error Function
                Optimizer.VANILLA,//Stochastic Gradient Descent Optimizer
                new Layer.Dense(2, 2, ActivationFunction.SIGMOID, Initializer.VANILLA),//2 in, 2 out
                new Layer.Dense(2, 1, ActivationFunction.SIGMOID, Initializer.VANILLA)//2 in from the previous layer, 1 out
        );
        System.out.println(nn.NETWORKSIZE);
        System.out.println(nn.toString());
        ArrayList<Data> data = new ArrayList<>();

        //XOR truth table
        data.add(new Data(new float[]{1, 1}, new float[]{0}));//True, True = False
        data.add(new Data(new float[]{0, 1}, new float[]{1}));//False, True = True
        data.add(new Data(new float[]{1, 0}, new float[]{1}));//True, False = True
        data.add(new Data(new float[]{0, 0}, new float[]{0}));//False, False = False
        NNLib.showInfo(NNLib.INFO_GRAPH(false), nn);
        NNLib.showInfo(NNLib.INFO_GRAPH(true), nn);
        for (int i = 0; i < 1_000_000_000; i++) {
            int index = nn.getRandom().nextInt(4);
            if (PRINT && i % 100000 == 0) {
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
