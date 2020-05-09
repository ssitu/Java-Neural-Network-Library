package neuralnetwork;

import java.util.ArrayList;
import java.util.Random;
import neuralnetwork.NNLib.*;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        long seed = new Random().nextLong();
        NN nn = new NNLib().new NN(
                "XOR",//Name for Saving & Graph Title
                seed,//Seed For Reproducibility
                .01,//Learning Rate for Optimizer
                LossFunction.QUADRATIC(.5),//Loss/Cost/Error Function
                Optimizer.VANILLA,//Stochastic Gradient Descent Optimizer
                new Layer.Dense(2, 3, ActivationFunction.SIGMOID, Initializer.VANILLA),//2 in, 3 out, using 3 nodes in the hidden layer for guaranteed convergence
                new Layer.Dense(3, 1, ActivationFunction.SIGMOID, Initializer.VANILLA)//3 in from the previous layer, 1 out
        );//Minimalistic network gets stuck a lot
        System.out.println("Seed: " + seed);
        System.out.println("Network Length: " + nn.length);
        System.out.println("Network Architecture: " + nn.toString());
        ArrayList<Data> data = new ArrayList<>();
        nn.load();

        //XOR truth table
        data.add(new Data(new float[]{1, 1}, new float[]{0}));//True, True = False
        data.add(new Data(new float[]{0, 1}, new float[]{1}));//False, True = True
        data.add(new Data(new float[]{1, 0}, new float[]{1}));//True, False = True
        data.add(new Data(new float[]{0, 0}, new float[]{0}));//False, False = False
        NNLib.setInfoUpdateRate(10);
        NNLib.showInfo(NNLib.infoLayers, nn);
        NNLib.showInfo(NNLib.infoGraph(true), nn);
        NNLib.showInfo(NNLib.infoGraph(false), nn);
        for (int i = 0; i < 100_000_000; i++) {
            int index = nn.getRandom().nextInt(4);
            if (PRINT && i % 100_000 == 0) {
                System.out.println("Inputs:");
                NNLib.print(data.get(index).inputs);
                System.out.println("Outputs:");
                NNLib.print(nn.feedforward(data.get(index).inputs));
                System.out.println("");
                nn.save();
            }
            nn.backpropagation(data.get(index).inputs, data.get(index).targets);
        }
//        System.exit(0);
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
