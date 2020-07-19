package testcases;

import nnlibrary.NNlib.*;
import static nnlibrary.NNlib.*;
import nnlibrary.NNlib.Layer.*;

public class Benchmarking {

    public static void main(String[] args) {
        NN nn = new NN("numberclassifier", 0, 0, LossFunction.CROSSENTROPY(1), Optimizer.ADADELTA,
                new Conv(10, 1, 3, 3, 1, a -> a, Activation.LEAKYRELU),//1x28x28 conv(s=1) 10x1x3x3 = 10x26x26
                new Conv(7, 10, 4, 4, 2, a -> a, Activation.LEAKYRELU),//10x26x26 conv(s=2) 7x10x4x4 = 7x12x12
                new Conv(4, 7, 5, 5, 1, a -> a, Activation.LEAKYRELU),//7x12x12 conv(s=1) 4x7x5x5 = 4x8x8
                new Conv(1, 4, 6, 6, 1, a -> a, Activation.LEAKYRELU),//4x8x8 conv(s=1) 1x4x6x6 = 1x3x3
                new Flatten(1, 3, 3),//1x3x3 = 9
                new Dense(9, 10, Activation.SOFTMAX, Initializer.XAVIER)
        );
        float[][][] input = function2dOn3d(new float[1][28][28], a -> randomize(a, 2, -1, nn.getRandom()));
        float[][] label = {{1}};
        System.out.println(benchmark(() -> nn.backpropagation(input, label), 1000000, false));
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
