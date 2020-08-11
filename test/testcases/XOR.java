package testcases;

import nnlibrary.NNlib;
import nnlibrary.NNlib.*;
import java.util.Random;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        final int BATCHSIZE = 4;
        long seed = new Random().nextLong();
        NN nn = new NN(
                "XOR",//Name for Saving & Graph Title
                seed,//Seed For Reproducibility
                .1f,//Learning Rate for Optimizer
                LossFunction.QUADRATIC(.5),//Loss/Cost/Error Function
                Optimizer.VANILLA,//Gradient Descent Optimizer
                new Layer.Dense(2, 3, Activation.SIGMOID, Initializer.VANILLA),//2 in, 3 out
                new Layer.Dense(1, Activation.SIGMOID, Initializer.VANILLA)//3 in from the previous layer, 1 out
        );
        nn.setBatchSize(BATCHSIZE);
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
        NNlib.setInfoUpdateRate(10);//The info panels will update every 10 milliseconds
        NNlib.showInfo(NNlib.infoLayers, nn);//Displays the weights and biases of each layer. Can be very intensive if used on a large network with a fast update rate
        NNlib.showInfo(NNlib.infoGraph(true), nn);//Displays an accuracy over number of times backpropgated graph
        NNlib.showInfo(NNlib.infoGraph(false), nn);//Displays a cost over number of times backpropagated graph
        for (int i = 0; i < 100_000_000; i++) {
            if (BATCHSIZE == 1) {
//                //Get a random data pair with the NN's seed
                int index = nn.getRandom().nextInt(4);
                nn.backpropagation(dataset[0][index], dataset[1][index]);//Tunes network parameters to output values closer to the labels given the inputs.
            } else {
                if (i % 4 == 0) {
                    nn.backpropagation(dataset[0][0], dataset[1][0]);
                } else if (i % 4 == 1) {
                    nn.backpropagation(dataset[0][1], dataset[1][1]);
                } else if (i % 4 == 2) {
                    nn.backpropagation(dataset[0][2], dataset[1][2]);
                } else if (i % 4 == 3) {
                    nn.backpropagation(dataset[0][3], dataset[1][3]);
                }
            }
            if (PRINT && i % 100000 == 0) {
                System.out.println("XOR Inputs:");
                float[][] inputs = NNlib.append(dataset[0][0], dataset[0][1], dataset[0][2], dataset[0][3]);
                NNlib.print(inputs);
                System.out.println("Outputs:");
                NNlib.print((float[][]) nn.feedforward(dataset[0][0]));
                NNlib.print((float[][]) nn.feedforward(dataset[0][1]));
                NNlib.print((float[][]) nn.feedforward(dataset[0][2]));
                NNlib.print((float[][]) nn.feedforward(dataset[0][3]));
                System.out.println("");
            }
        }
        System.exit(0);
    }
}
