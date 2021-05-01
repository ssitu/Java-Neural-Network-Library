package testcases;

import static nnlibrary.hyperparameters.Functions.*;
import nnlibrary.*;
import static nnlibrary.JavaFXTools.*;
import nnlibrary.hyperparameters.*;
import nnlibrary.layers.*;

public class Benchmarking {

    public static void main(String[] args) {
        NN nn = new NN("numberclassifier", 0, 0, LossFunctions.CROSSENTROPY(1), Optimizers.ADADELTA,
                new Conv(1, 28, 28, 10, 3, 3, 1, 0, 0, Activations.LEAKYRELU),//1x28x28 conv(s=1) 10x1x3x3 = 10x26x26
                new Conv(7, 4, 4, 2, 0, 0, Activations.LEAKYRELU),//10x26x26 conv(s=2) 7x10x4x4 = 7x12x12
                new Conv(4, 5, 5, 1, 0, 0, Activations.LEAKYRELU),//7x12x12 conv(s=1) 4x7x5x5 = 4x8x8
                new Conv(1, 6, 6, 1, 0, 0, Activations.LEAKYRELU),//4x8x8 conv(s=1) 1x4x6x6 = 1x3x3
                new Flatten(),//1x3x3 = 9
                new Dense(10, Activations.SOFTMAX, Initializers.XAVIER)
        );
        System.out.println(nn);
        nn.setAccumulationSize(100);
        showInfo(infoGraph(false), nn);
        float[][][] input = function2dOn3d(new float[1][28][28], a -> randomize(a, 2, -1, nn.getRandom()));
        float[][] label = {{1}};
        System.out.println("Seconds: " + benchmark(() -> nn.backpropagation(input, label), 1000000, false));
    }

    public static double benchmark(Runnable r, int iterations, boolean avg) {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            r.run();
        }
        long end = System.nanoTime();
        if (avg) {
            return .000000001 * (end - start) / iterations;
        } else {
            return .000000001 * (end - start);
        }
    }
}
