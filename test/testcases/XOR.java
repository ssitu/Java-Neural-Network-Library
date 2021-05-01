package testcases;

import java.util.Random;
import nnlibrary.*;
import static nnlibrary.JavaFXTools.*;
import nnlibrary.hyperparameters.*;
import nnlibrary.layers.*;

public class XOR {

    public static void main(String[] args) {
        final boolean PRINT = true;
        long seed = new Random().nextLong();
        NN nn = new NN(
                "XOR",//Name for Saving & Graph Title
                seed,//Seed For Reproducibility
                .1f,//Learning Rate for Optimizer
                LossFunctions.QUADRATIC(.5),//Loss/Cost/Error Function
                Optimizers.VANILLA,//Gradient Descent Optimizer
                new Dense(2, 3, Activations.SIGMOID, Initializers.VANILLA),//2 in, 3 out
                new Dense(1, Activations.SIGMOID, Initializers.VANILLA)//3 in from the previous layer, 1 out
        );
        nn.setAccumulationSize(4);
        System.out.println("Seed: " + seed);
        System.out.println("Network Length: " + nn.length);
        System.out.println("Network Architecture: " + nn);
        System.out.println("Network Parameters: " + nn.getParameterCount());
        System.out.println("====================================================");
        System.out.println("Network Starting Parameters: ");
        for(int i = 0; i < nn.length; i++){
            System.out.println("Layer " + i + ":");
            System.out.println(nn.getLayer(i).parametersToString());
        }
        System.out.println("====================================================");
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
        setInfoUpdateRate(10);//The info panels will update every 10 milliseconds
        showInfo(infoLayers, nn);//Displays the weights and biases of each layer. Can be very intensive if used on a large network with a fast update rate
        showInfo(infoGraph(true), nn);//Displays an accuracy over number of times backpropgated graph
        showInfo(infoGraph(false), nn);//Displays a cost over number of times backpropagated graph
        for (int i = 0; i < 100_000_000; i++) {
            int index = i % 4;
            nn.backpropagation(dataset[0][index], dataset[1][index]);//Tunes network parameters to output values closer to the labels given the inputs.
            if (PRINT && i % 100000 == 0) {
                System.out.println("XOR Inputs:");
                float[][] inputs = Functions.append(dataset[0][0], dataset[0][1], dataset[0][2], dataset[0][3]);
                Functions.print(inputs);
                System.out.println("Outputs:");
                Functions.print((float[][]) nn.feedforward(dataset[0][0]));
                Functions.print((float[][]) nn.feedforward(dataset[0][1]));
                Functions.print((float[][]) nn.feedforward(dataset[0][2]));
                Functions.print((float[][]) nn.feedforward(dataset[0][3]));
                System.out.println("");
            }
        }
        System.exit(0);
    }
}
