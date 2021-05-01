package nnlibrary;

import static nnlibrary.hyperparameters.Functions.*;

public class Init {

    public static final long serial = 0;
    public static final int usableProcessors = Runtime.getRuntime().availableProcessors() / 4;
    public static BiFunction<float[][], float[][], float[][]> dotProduct = (a, b) -> {
        if (a.length == 1) {
            return dotVector(a[0], b);
//        } else if (a.length / usableProcessors > 10) {
//            return dotThreads(a, b, usableProcessors);
        } else {
            return dot(a, b);
        }
    };

}
