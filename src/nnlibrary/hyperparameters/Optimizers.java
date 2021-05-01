package nnlibrary.hyperparameters;

import java.io.Serializable;
import static nnlibrary.hyperparameters.Functions.abs;
import static nnlibrary.hyperparameters.Functions.add;
import static nnlibrary.hyperparameters.Functions.create;
import static nnlibrary.hyperparameters.Functions.divide;
import static nnlibrary.hyperparameters.Functions.ewma;
import static nnlibrary.hyperparameters.Functions.max;
import static nnlibrary.hyperparameters.Functions.multiply;
import static nnlibrary.hyperparameters.Functions.scale;
import static nnlibrary.hyperparameters.Functions.sqrt;
import static nnlibrary.hyperparameters.Functions.square;

/**
     * A set of common optimizers. The first three parameters of the
     * QuadFunction are strictly for the time step, learning rate, and the
     * gradients respectively. The third parameter is an array of matrices to
     * store info such as previous updates, gradients, etc. The QuadFunction
     * returns an array where its first element is a matrix with the gradient
     * update and its second element is the storage array(3 dimensional) that is
     * passed into the next call of the optimizer.
     */
    public class Optimizers {

        public static final float beta = .9f;
        public static final float beta2 = .999f;
        public static final float e = 1e-7f;

        public interface Optimizer extends Serializable {

            Object[] apply(int step, float learningRate, float[][] gradients, float[][][] storage);
        }
        public static Optimizer VANILLA = (step, lr, gradients, storage) -> {
            float[][] update = scale(lr, gradients);
            return new Object[]{update, null};
        };
        public static Optimizer MOMENTUM = (step, lr, gradients, storage) -> {
            float[][] update = add(scale(beta, storage[0]), scale(lr, gradients));
            float[][][] store = {update};
            return new Object[]{update, store};
        };
        public static Optimizer NESTEROV = (step, lr, gradients, storage) -> {//Dozat's modification because calculating gradients of different parameters would take a lot of reworking
            float[][] m = add(scale(beta, storage[0]), gradients);
            float[][] update = scale(lr, add(gradients, scale(beta, m)));
            float[][][] store = {m};
            return new Object[]{update, store};
        };
        public static Optimizer ADAGRAD = (step, lr, gradients, storage) -> {
            float[][][] store = {add(storage[0], square(gradients))};
            float[][] update = multiply(scale(lr, sqrt(add(store[0], create(gradients.length, gradients[0].length, e)))), gradients);
            return new Object[]{update, store};
        };
        public static Optimizer ADADELTA = (step, lr, gradients, storage) -> {
            float[][] epsilon = create(gradients.length, gradients[0].length, e);
            float[][] gradientsE = ewma(beta, storage[0], square(gradients));
            float[][] gradientsRMS = sqrt(add(gradientsE, epsilon));
            float[][] deltaRMS = sqrt(add(storage[1], epsilon));
            float[][] update = multiply(divide(deltaRMS, gradientsRMS), gradients);
            float[][] deltaE = ewma(beta, storage[1], square(update));
            float[][][] store = {gradientsE, deltaE};
            return new Object[]{update, store};
        };
        public static Optimizer RMSPROP = (step, lr, gradients, storage) -> {
            float[][] s = ewma(beta, storage[0], square(gradients));
            float[][] s_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), s);
            float[][] update = divide(scale(lr, gradients), add(sqrt(s_), create(s.length, s[0].length, e)));
            float[][][] store = {s};
            return new Object[]{update, store};
        };
        public static Optimizer ADAM = (step, lr, gradients, storage) -> {
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] update = divide(scale(lr, m_), add(sqrt(v_), create(v_.length, v_[0].length, e)));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
        public static Optimizer ADAMAX = (step, lr, gradients, storage) -> {
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = multiply(m, 1 / (1 - (float) Math.pow((double) beta, step)));
            float[][] u = max(scale(beta2, storage[1]), abs(gradients));
            float[][] update = divide(scale(lr, m_), add(u, create(u.length, u[0].length, e)));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
        public static Optimizer NADAM = (step, lr, gradients, storage) -> {
            int rows = gradients.length;
            int columns = gradients[0].length;
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] update = multiply(divide(create(rows, columns, lr), add(sqrt(v_), create(rows, columns, e))), add(scale(beta, m_), multiply(scale(1 - beta, gradients), 1 / (1 - beta))));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
        public static Optimizer AMSGRAD = (step, lr, gradients, storage) -> {
            float[][] m = ewma(beta, storage[0], gradients);
            float[][] v = ewma(beta2, storage[1], square(gradients));
            float[][] m_ = scale(1 / (1 - (float) Math.pow((double) beta, step)), m);
            float[][] v_ = scale(1 / (1 - (float) Math.pow((double) beta2, step)), v);
            float[][] v__ = max(storage[1], v_);
            float[][] update = divide(scale(lr, m_), add(sqrt(v__), create(v__.length, v__[0].length, e)));
            float[][][] store = {m, v};
            return new Object[]{update, store};
        };
    }