package neuralnetwork;

import neuralnetwork.NNLib.*;
import java.util.ArrayList;
import java.util.Random;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        long seed = new Random().nextLong();
        NN nn = new NN(//0.077241674
                "XOR",//Name for Saving & Graph Title
                seed,//Seed For Reproducibility
                .01,//Learning Rate for Optimizer
                LossFunction.QUADRATIC(.5),//Loss/Cost/Error Function
                Optimizer.VANILLA,//Stochastic Gradient Descent Optimizer
                new Layer.Dense(2, 3, ActivationFunction.SIGMOID, Initializer.VANILLA),//2 in, 3 out
                new Layer.Dense(3, 1, ActivationFunction.SIGMOID, Initializer.VANILLA)//3 in from the previous layer, 1 out
        );//The classic 2_2_1 network gets stuck a lot, adding an extra node in the hidden layer guarantees convergence
        System.out.println("Seed: " + seed);
        System.out.println("Network Length: " + nn.length);
        System.out.println("Network Architecture: \n" + nn);
        ArrayList<Data> data = new ArrayList<>();

        //XOR truth table
        data.add(new Data(new float[][]{{1, 1}}, new float[][]{{0}}));//T T = F
        data.add(new Data(new float[][]{{0, 1}}, new float[][]{{1}}));//F T = T
        data.add(new Data(new float[][]{{1, 0}}, new float[][]{{1}}));//T F = T
        data.add(new Data(new float[][]{{0, 0}}, new float[][]{{0}}));//F F = F
        NNLib.setInfoUpdateRate(10);//The info panels will update every 10 milliseconds
        NNLib.showInfo(NNLib.infoLayers, nn);//Displays the weights and biases of each layer. Can be very intensive if used on a large network with a fast update rate
        NNLib.showInfo(NNLib.infoGraph(true), nn);//Displays an accuracy over number of times backpropgated graph
        NNLib.showInfo(NNLib.infoGraph(false), nn);//Displays a cost over number of times backpropagated graph
        for (int i = 0; i < 100_000_000; i++) {
            //Get a random data pair
            int index = nn.getRandom().nextInt(4);
            Data sample = data.get(index);
            if (PRINT && i % 100000 == 0) {
                System.out.println("Inputs:");
                NNLib.print(sample.inputs);
                float[][] forwardPass = nn.feedforward(sample.inputs);
                System.out.println("Outputs:");
                NNLib.print(forwardPass);
                System.out.println("");
            }
            nn.backpropagation(sample.inputs, sample.labels);//Trains the network to output the targets with the given inputs, affects network parameters
        }
        System.exit(0);
    }

    static class Data {//Simple pair of inputs and labels

        float[][] inputs;
        float[][] labels;

        Data(float[][] inputs, float[][] labels) {
            this.inputs = inputs;
            this.labels = labels;
        }
    }
}
