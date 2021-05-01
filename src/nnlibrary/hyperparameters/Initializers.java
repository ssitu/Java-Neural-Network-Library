package nnlibrary.hyperparameters;

import java.io.Serializable;
import static nnlibrary.hyperparameters.Functions.multiply;

/**
     * A set of common weight initializers. Takes in weights of a layer and the
     * number of nodes going into the layer ()
     */
    public class Initializers {

        /**
         * A functional interface for the initialization of parameters.
         */
        public interface Initializer extends Serializable {

            /**
             * Applies the initializer to the given parameters.
             *
             * @param parameters A 2d array of parameters for the initializer to
             * modify.
             * @param nodesIn The nodes going into the Layer with the
             * parameters.
             * @return A 2d array with the new parameters.
             */
            float[][] apply(float[][] parameters, int nodesIn);
        }
        public static final Initializer VANILLA = (a, b) -> a;//No change
        public static final Initializer XAVIER = (a, b) -> multiply(a, (float) Math.sqrt(1.0 / b));
        public static final Initializer HE = (a, b) -> multiply(a, (float) Math.sqrt(2.0 / b));
    }