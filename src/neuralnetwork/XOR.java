package neuralnetwork;

import neuralnetwork.NNLib.*;
import java.util.Random;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        final int MINIBATCHES = 4;//4 = Stochastic Gradient Descent, 2 = Minibatches with the dataset cut in half, 1 = Batch Gradient Descent
        long seed = new Random().nextLong();
        NN nn = new NN(
                "XOR",//Name for Saving & Graph Title
                seed,//Seed For Reproducibility
                .01f,//Learning Rate for Optimizer
                LossFunction.QUADRATIC(.5),//Loss/Cost/Error Function
                Optimizer.VANILLA,//Gradient Descent Optimizer
                new Layer.Dense(2, 2, ActivationFunction.SIGMOID, Initializer.VANILLA),//2 in, 3 out
                new Layer.Dense(2, 1, ActivationFunction.SIGMOID, Initializer.VANILLA)//3 in from the previous layer, 1 out
        );
        System.out.println("Seed: " + seed);
        System.out.println("Network Length: " + nn.length);
        System.out.println("Network Architecture: \n" + nn);
        //XOR truth table
        float[][][][] dataset = {
            {//Inputs
                {{0, 0}},//F F
                {{0, 1}},//F T
                {{1, 0}},//T F
                {{1, 1}}//T T
            },
            {//Labels
                {{0}},//F
                {{1}},//T
                {{1}},//T
                {{0}}//F
            }
        };
        NNLib.setInfoUpdateRate(10);//The info panels will update every 10 milliseconds
        NNLib.showInfo(NNLib.infoLayers, nn);//Displays the weights and biases of each layer. Can be very intensive if used on a large network with a fast update rate
        NNLib.showInfo(NNLib.infoGraph(true), nn);//Displays an accuracy over number of times backpropgated graph
        NNLib.showInfo(NNLib.infoGraph(false), nn);//Displays a cost over number of times backpropagated graph
        for (int i = 0; i < 100_000_000; i++) {
            if (MINIBATCHES == 4) {//Stochastic
                //Get a random data pair with the NN's seed
                int index = nn.getRandom().nextInt(4);
                nn.backpropagation(dataset[0][index], dataset[1][index]);//Tunes network parameters to output values closer to the labels given the inputs.
            } else if (MINIBATCHES == 2) {
                if (i % 2 == 0) {//First minibatch
                    nn.backpropagation(
                            NNLib.append(dataset[0][0], dataset[0][1]),//Inputs
                            NNLib.append(dataset[1][0], dataset[1][1]));//Labels
                } else {//Second minibatch
                    nn.backpropagation(
                            NNLib.append(dataset[0][2], dataset[0][3]),//Inputs
                            NNLib.append(dataset[1][2], dataset[1][3]));//Labels
                }
            } else if (MINIBATCHES == 1) {//Full dataset
                nn.backpropagation(
                        NNLib.append(dataset[0][0], dataset[0][1], dataset[0][2], dataset[0][3]),//Inputs
                        NNLib.append(dataset[1][0], dataset[1][1], dataset[1][2], dataset[1][3]));//Labels
            }
            if (PRINT && i % 100000 == 0) {
                System.out.println("XOR Inputs:");
                float[][] inputs = NNLib.append(dataset[0][0], dataset[0][1], dataset[0][2], dataset[0][3]);//The whole dataset's inputs
                NNLib.print(inputs);
                float[][] forwardPass = nn.feedforward(inputs);
                System.out.println("Outputs:");
                NNLib.print(forwardPass);
                System.out.println("");
            }
        }
        System.exit(0);
    }
}
