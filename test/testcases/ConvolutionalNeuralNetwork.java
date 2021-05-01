package testcases;

import java.util.Random;
import static nnlibrary.hyperparameters.Functions.*;
import static nnlibrary.JavaFXTools.*;
import nnlibrary.*;
import nnlibrary.hyperparameters.*;
import nnlibrary.layers.*;

public class ConvolutionalNeuralNetwork {

    public static void main(String[] args) {
        long seed = new Random().nextLong();
        System.out.println("Seed: " + seed);
        NN nn = new NN("Conv", seed, 0, LossFunctions.QUADRATIC(.5), Optimizers.ADADELTA,
                new Conv(1, 3, 3, 2, 3, 3, 1, 1, 1, Activations.TANH),//1x3x3 + 1x1 pad = 1x5x5, 1x5x5 conv(s=1) 2x1x3x3 = 2x3x3
                new Maxpool(2, 2, 1),//2x3x3 maxpool(s=1) 2x2 = 2x2x2
                new Conv(1, 1, 1, 1, 0, 0, Activations.TANH),//2x2x2 conv(s=1) 1x2x1x1 = 1x2x2
                new Flatten(),//1x2x2 = 4
                new Dense(1, Activations.SIGMOID, Initializers.XAVIER));
        showInfo(infoLayers, nn);
        showInfo(infoGraph(false), nn);
        float[][][] input1 = {//Slash
            {
                {0, 0, 1},
                {0, 1, 0},
                {1, 0, 0}
            }
        };
        float[][] label1 = {{1}};
        float[][][] input2 = {//Slash
            {
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
            }
        };
        float[][] label2 = {{1}};
        float[][][] input3 = {//Not a slash
            {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            }
        };
        float[][] label3 = {{0}};
        float[][][] input4 = {//Not a slash
            {
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0}
            }
        };
        float[][] label4 = {{0}};
        for (int i = 0; i < 10000000; i++) {
            nn.backpropagation(input1, label1);
            nn.backpropagation(input2, label2);
            nn.backpropagation(input3, label3);
            nn.backpropagation(input4, label4);
            if (i % 10000 == 0) {
                System.out.println("Inputs:");
                print(input1);
                print(input2);
                print(input3);
                print(input4);
                System.out.println("Labels:");
                print(label1);
                print(label2);
                print(label3);
                print(label4);
                System.out.println("Outputs:");
                print((float[][]) nn.feedforward(input1));
                print((float[][]) nn.feedforward(input2));
                print((float[][]) nn.feedforward(input3));
                print((float[][]) nn.feedforward(input4));
            }
        }
    }
}
