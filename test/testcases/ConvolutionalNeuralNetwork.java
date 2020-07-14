package testcases;

import java.util.Random;
import static nnlibrary.NNLib.*;
import nnlibrary.NNLib.*;
import nnlibrary.NNLib.Layer.*;

public class ConvolutionalNeuralNetwork {

    public static void main(String[] args) {
        NN nn = new NN("Conv", new Random().nextLong(), 0, LossFunction.QUADRATIC(.5), Optimizer.ADADELTA,
                new Conv(1, 1, 2, 2, 0, 1, Activation.RELU),
                new Flatten(1, 2, 2),
                new Dense(4, 1, Activation.SIGMOID, Initializer.XAVIER));
        float[][][] input1 = {
            {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            }
        };
        float[][] label1 = {{1}};
        float[][][] input2 = {
            {
                {0, 1, 0},
                {1, 0, 1},
                {0, 1, 0}
            }
        };
        float[][] label2 = {{1}};
        for (int i = 0; i < 10000000; i++) {
            nn.backpropagation(input1, label1);
            nn.backpropagation(input2, label2);
            if (i % 10000 == 0) {
                print3d(input1, "input 1");
                print3d(input2, "input 2");
                print((float[][]) nn.feedforward(input1));
                print((float[][]) nn.feedforward(input2));
            }
        }
    }
}
