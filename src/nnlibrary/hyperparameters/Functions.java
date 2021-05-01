package nnlibrary.hyperparameters;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Random;

public class Functions {
    
    /**
     * A serializable version of java's Function interface
     */
    public static interface Function<T, R> extends java.util.function.Function<T, R>, Serializable {
    }

    /**
     * A serializable version of java's BiFunction interface
     */
    public static interface BiFunction<T, S, R> extends java.util.function.BiFunction<T, S, R>, Serializable {
    }
    
    public static float[][] normalizeMinMax(float[][] oneRow) {
        int elements = oneRow[0].length;
        float[][] result = new float[1][elements];
        float min = min(oneRow);
        float max = max(oneRow);
        for (int i = 0; i < elements; i++) {
            result[0][i] = (oneRow[0][i] - min) / (max - min);
        }
        return result;
    }

    public static float[][] normalizeZScore(float[][] oneRow) {
        int elements = oneRow[0].length;
        float mean = sum(oneRow) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(oneRow, create(1, elements, mean)))) / (mean)));
        return divide(subtract(oneRow, create(1, elements, mean)), create(1, elements, deviation));
    }

    public static float[][] normalizeTanh(float[][] oneRow) {
        int elements = oneRow[0].length;
        float[][] result = new float[1][elements];
        float mean = sum(oneRow) / elements;
        float deviation = (float) (Math.sqrt(sum(square(subtract(oneRow, create(1, elements, mean)))) / mean));
        for (int i = 0; i < elements; i++) {
            result[0][i] = (.5f * (tanh((.01f * ((oneRow[0][i] - mean) / (deviation))), false) + 1));
        }
        return result;
    }

    public static float sigmoid(float x, boolean derivative) {
        if (!derivative) {
            return 1 / (1 + (float) Math.exp(-x));//sigmoid(x)
        } else {
            return sigmoid(x, false) * (1 - sigmoid(x, false));//sigmoid'(x)
        }
    }

    public static float tanh(float x, boolean derivative) {
        if (!derivative) {
            return (2 / (1 + (float) Math.exp(-2 * x))) - 1;//tanh(x)
        } else {
            float val = tanh(x, false);
            return 1 - val * val;//tanh'(x)
        }
    }

    public static float relu(float x, boolean derivative) {
        if (!derivative) {
            return Math.max(0, x);
        } else {
            if (x < 0) {
                return 0;
            }
            return 1;
        }
    }

    public static float leakyrelu(float x, boolean derivative) {
        if (!derivative) {
            return Math.max(.001f * x, x);
        } else {
            if (x < 0) {
                return .001f;
            }
            return 1;
        }
    }

    public static float swish(float x, boolean derivative) {
        if (!derivative) {
            return x * sigmoid(x, false);
        } else {
            return x * sigmoid(x, true) + sigmoid(x, false);
        }
    }

    public static float mish(float x, boolean derivative) {
        if (!derivative) {
            return (x * tanh((float) Math.log(1 + Math.exp(x)), false));
        } else {
            double x_ = (double) x;
            return (float) ((Math.exp(x_) * ((4 * (x_ + 1)) + (4 * Math.exp(2 * x_)) + (Math.exp(3 * x_)) + (Math.exp(x_) * (4 * x_ + 6)))) / ((2 * Math.exp(2 * x_)) + (Math.exp(2 * x_)) + 2));
        }
    }

    /**
     * Performs softmax on each of the rows of the given matrix. Includes
     * numeric stability.
     *
     * @param matrix An array holding arrays of values where softmax is
     * performed on each array of values within the array.
     * @return The resulting matrix with the same shape as the given matrix.
     */
    public static float[][] softmax(float[][] matrix) {
        int cols = matrix[0].length;
        float[][] result = functionMatrixVectors(matrix, vector -> subtract(vector, create(1, cols, max(vector))));//Stabilizing
        result = exp(result);
        return functionMatrixVectors(result, vector -> divide(vector, create(1, cols, sum(vector))));
    }

    private static class DotWorker extends Thread {

        int threadNum;
        int totalThreads;
        int rows1;
        int cols1;
        int cols2;
        float[][] m1;
        float[][] m2;
        float[][] result;

        DotWorker(int threadNum, int totalThreads, int rows1, int cols1, int cols2, float[][] m1, float[][] m2, float[][] result) {
            this.threadNum = threadNum;
            this.totalThreads = totalThreads;
            this.rows1 = rows1;
            this.cols1 = cols1;
            this.cols2 = cols2;
            this.m1 = m1;
            this.m2 = m2;
            this.result = result;
        }

        @Override
        public void run() {
            int endRow = (threadNum + 1) * rows1 / totalThreads;
            if (cols1 % 4 == 0) {//Loop unrolling increases speed
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k += 4) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j]
                                    + m1[i][k + 1] * m2[k + 1][j]
                                    + m1[i][k + 2] * m2[k + 2][j]
                                    + m1[i][k + 3] * m2[k + 3][j];
                        }
                    }
                }
            } else if (cols1 % 3 == 0) {
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k += 3) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j]
                                    + m1[i][k + 1] * m2[k + 1][j]
                                    + m1[i][k + 2] * m2[k + 2][j];
                        }
                    }
                }
            } else if (cols1 % 2 == 0) {
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k += 2) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j]
                                    + m1[i][k + 1] * m2[k + 1][j];
                        }
                    }
                }
            } else {
                for (int i = threadNum * rows1 / totalThreads; i < endRow; i++) {
                    for (int k = 0; k < cols1; k++) {
                        for (int j = 0; j < cols2; j++) {
                            result[i][j] += m1[i][k] * m2[k][j];
                        }
                    }
                }
            }
        }
    }

    public static float[][] dotThreads(float[][] m1, float[][] m2, int threads) {
        DotWorker[] threadArray = new DotWorker[threads];
        int rows1 = m1.length;
        int cols1 = m1[0].length;
        int cols2 = m2[0].length;
        float[][] result = new float[rows1][cols2];
        for (int t = 0; t < threads; t++) {
            threadArray[t] = new DotWorker(t, threads, rows1, cols1, cols2, m1, m2, result);
            threadArray[t].start();
        }
        for (int i = 0; i < threads; i++) {
            try {
                threadArray[i].join();
            } catch (InterruptedException e) {
            }
        }
        return result;
    }

    public static float[][] dot(float[][] m1, float[][] m2) {
        int rows1 = m1.length;
        int cols1 = m1[0].length;
        int cols2 = m2[0].length;
        float[][] result = new float[rows1][cols2];
        if (cols1 % 4 == 0) {//Loop unrolling increases speed
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k += 4) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j]
                                + m1[i][k + 2] * m2[k + 2][j]
                                + m1[i][k + 3] * m2[k + 3][j];
                    }
                }
            }
        } else if (cols1 % 3 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k += 3) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j]
                                + m1[i][k + 2] * m2[k + 2][j];
                    }
                }
            }
        } else if (cols1 % 2 == 0) {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k += 2) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j]
                                + m1[i][k + 1] * m2[k + 1][j];
                    }
                }
            }
        } else {
            for (int i = 0; i < rows1; i++) {
                for (int k = 0; k < cols1; k++) {
                    for (int j = 0; j < cols2; j++) {
                        result[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
        }
        return result;
    }

    public static float[][] dotVector(float[] vector, float[][] matrix) {
        int rows2 = matrix.length;
        int cols2 = matrix[0].length;
        float[][] result = new float[1][cols2];
        if (rows2 % 4 == 0) {
            for (int i = 0; i < rows2; i += 4) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j]
                            + vector[i + 1] * matrix[i + 1][j]
                            + vector[i + 2] * matrix[i + 2][j]
                            + vector[i + 3] * matrix[i + 3][j];
                }
            }
        } else if (rows2 % 3 == 0) {
            for (int i = 0; i < rows2; i += 3) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j]
                            + vector[i + 1] * matrix[i + 1][j]
                            + vector[i + 2] * matrix[i + 2][j];
                }
            }
        } else if (rows2 % 2 == 0) {
            for (int i = 0; i < rows2; i += 2) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j]
                            + vector[i + 1] * matrix[i + 1][j];
                }
            }
        } else {
            for (int i = 0; i < rows2; i++) {
                for (int j = 0; j < cols2; j++) {
                    result[0][j] += vector[i] * matrix[i][j];
                }
            }
        }
        return result;
    }

    public static String arrToString(float[][] arr2d) {
        int rowLastIndex = arr2d.length - 1;
        int columns = arr2d[0].length;
        String string = "[";
        for (int i = 0; i < rowLastIndex; i++) {
            string += "[" + arr2d[i][0];
            for (int j = 1; j < columns; j++) {
                string += " " + arr2d[i][j];
            }
            string += "]\n";
        }
        string += "[" + arr2d[rowLastIndex][0];
        for (int j = 1; j < columns; j++) {
            string += " " + arr2d[rowLastIndex][j];
        }
        string += "]]\n";
        return string;
    }

    public static String arrToString(float[][][] arr3d) {
        int depth = arr3d.length;
        String string = "[\n" + arrToString(arr3d[0]);
        for (int i = 1; i < depth; i++) {
            string += "\n" + arrToString(arr3d[i]);
        }
        return string + "]\n";
    }

    public static void print(float[][] arr2d) {
        System.out.print(arrToString(arr2d));
    }

    public static void print(float[][] arr2d, String label) {
        System.out.println(label + ":");
        print(arr2d);
    }

    public static void print(float[][][] arr3d) {
        int depth = arr3d.length;
        System.out.println("[");
        System.out.print(arrToString(arr3d[0]));
        for (int i = 1; i < depth; i++) {
            System.out.print("\n" + arrToString(arr3d[i]));
        }
        System.out.println("]");
    }

    public static void print(float[][][] arr3d, String label) {
        System.out.println(label + ":");
        print(arr3d);
    }

    public static float[][] doubleToFloat(double[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = (float) (matrix[i][j]);
            }
        }
        return result;
    }

    public static float[] create(int columns, float valueToAllElements) {
        float[] result = new float[columns];
        for (int i = 0; i < columns; i++) {
            result[i] = valueToAllElements;
        }
        return result;
    }

    public static float[][] create(int rows, int columns, float valueToAllElements) {
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = valueToAllElements;
            }
        }
        return result;
    }

    public static float[][][] create(int depth, int rows, int columns, float valueToAllElements) {
        float[][][] result = new float[depth][][];
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < columns; k++) {
                    result[i][j][k] = valueToAllElements;
                }
            }
        }
        return result;
    }

    public static float[][] randomize(float[][] matrix, float range, float minimum) {
        return function(matrix, val -> (float) Math.random() * range + minimum);
    }

    public static float[][] randomize(float[][] matrix, float range, float minimum, Random random) {
        return function(matrix, val -> random.nextFloat() * range + minimum);
    }

    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = new float[columns][rows];
        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public static float[][] multiply(float[][] matrix, float factor) {
        return function(matrix, val -> factor * val);
    }

    public static float[][] scale(float factor, float[][] matrix) {
        return function(matrix, val -> factor * val);
    }

    public static float[][] add(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 + val2);
    }

    public static float[][] subtract(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 - val2);
    }

    public static float[][] multiply(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 * val2);
    }

    public static float[][] divide(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> val1 / val2);
    }

    public static float[][] power(float[][] matrix, double power) {
        return function(matrix, val -> (float) Math.pow(val, power));
    }

    public static float[][] square(float[][] matrix) {
        return function(matrix, val -> val * val);
    }

    public static float[][] sqrt(float[][] matrix) {
        return function(matrix, val -> (float) Math.sqrt(val));
    }

    public static float[][] inverse(float[][] matrix) {
        return function(matrix, val -> 1 / val);
    }

    public static float sum(float[][] matrix) {
        float sum = 0;
        int rows = matrix.length;
        int columns = matrix[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sum += matrix[i][j];
            }
        }
        return sum;
    }

    public static float[][] abs(float[][] matrix) {
        return function(matrix, val -> Math.abs(val));
    }

    public static float[][] exp(float[][] matrix) {
        return function(matrix, val -> (float) Math.exp(val));
    }

    public static float[][] ln(float[][] matrix) {
        return function(matrix, val -> (float) Math.log(val));
    }

    public static float[][] ewma(float beta, float[][] prevStep, float[][] currentStep) {
        return add(scale(beta, prevStep), scale(1 - beta, currentStep));
    }

    /**
     * Replaces the maximum value in the given matrix with a 1 and replace the
     * rest of the elements with 0s.
     *
     * @param matrix a matrix with a unique maximum, otherwise the first maximum
     * value is chosen.
     * @return The resulting matrix.
     */
    public static float[][] oneHot(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = create(rows, columns, 0);
        float max = matrix[0][0];
        int x = 0;
        int y = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (max < matrix[i][j]) {
                    max = matrix[i][j];
                    x = i;
                    y = j;
                }
            }
        }
        result[x][y] = 1;
        return result;
    }

    /**
     * Creates a new matrix with the given dimensions and with a one in the
     * given position and 0s in the rest.
     *
     * @param rows Height of the result.
     * @param cols Width of the result.
     * @param onehotRow The row of the 1 in the result.
     * @param onehotCol The column of the 1 in the result.
     * @return The resulting matrix.
     */
    public static float[][] oneHot(int rows, int cols, int onehotRow, int onehotCol) {
        float[][] result = new float[rows][cols];
        result[onehotRow][onehotCol] = 1;
        return result;
    }

    /**
     * Samples from the given probabilities. Works properly when probabilities
     * add up to 1
     *
     * @param probabilities Array of probabilities adding up to 1.
     * @return The chosen index. Depending on the probabilities, will return -1
     * if probabilities don't add up to 1 and it resulted in no determined
     * index.
     */
    public static int sampleProbabilities(float[] probabilities) {
        float random = (float) Math.random();
        int size = probabilities.length;
        for (int i = 0; i < size; i++) {
            random -= probabilities[i];
            if (random < 0) {
                return i;
            }
        }
        return -1;
    }

    /**
     * @param probabilities Array of probabilities adding up to 1.
     * @param rng Random class for seeded randomness
     * @return The chosen index. Depending on the probabilities, will return -1
     * if probabilities don't add up to 1 and it resulted in no determined
     * index.
     * @see #sampleProbabilities(float[])
     */
    public static int sampleProbabilities(float[] probabilities, Random rng) {
        float random = rng.nextFloat();
        int size = probabilities.length;
        for (int i = 0; i < size; i++) {
            random -= probabilities[i];
            if (random < 0) {
                return i;
            }
        }
        return -1;
    }

    public static int[] includeElements(int[] array, int... indeces) {
        int length = indeces.length;
        int[] newArray = new int[length];
        for (int i = 0; i < length; i++) {
            newArray[i] = array[indeces[i]];
        }
        return newArray;
    }

    public static float[][] min(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> Math.min(val1, val2));
    }

    public static float[][] max(float[][] matrix1, float[][] matrix2) {
        return bifunction(matrix1, matrix2, (val1, val2) -> Math.max(val1, val2));
    }

    public static float min(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float min = matrix[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val = matrix[i][j];
                if (val < min) {
                    min = val;
                }
            }
        }
        return min;
    }

    public static float max(float[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float max = matrix[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float val = matrix[i][j];
                if (val > max) {
                    max = val;
                }
            }
        }
        return max;
    }

    public static int argmax(float[][] oneRowMatrix) {
        float max = Float.NEGATIVE_INFINITY;
        int index = 0;
        for (int i = 0; i < oneRowMatrix[0].length; i++) {
            if (max < oneRowMatrix[0][i]) {
                max = oneRowMatrix[0][i];
                index = i;
            }
        }
        return index;
    }

    public static int argmin(float[][] oneRowMatrix) {
        float min = Float.POSITIVE_INFINITY;
        int index = 0;
        for (int i = 0; i < oneRowMatrix[0].length; i++) {
            if (min > oneRowMatrix[0][i]) {
                min = oneRowMatrix[0][i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Creates a new array with the elements inside the given arrays in their
     * order. Arrays must have at least 1 element because empty arrays give a
     * length of 1 and the Class of the generic is obtained from the first
     * element from the first array given.
     *
     * @param <T> Object type.
     * @param arrays to be combined
     * @return The new extended array.
     */
    public static <T> T[] append(T[]... arrays) {
        int totalLength = arrays[0].length;
        int numOfArrays = arrays.length;
        for (int i = 1; i < numOfArrays; i++) {
            totalLength += arrays[i].length;
        }
        T[] result = (T[]) Array.newInstance(arrays[0][0].getClass(), totalLength);
        int index = 0;
        for (int i = 0; i < numOfArrays; i++) {
            T[] array = arrays[i];
            int length = array.length;
            for (int j = 0; j < length; j++) {
                result[index] = array[j];
                index++;
            }
        }
        return result;
    }

    public static int getDimensionality(Object[] arr) {
        int dimensions = 0;
        Class type = arr.getClass();
        while (type.isArray()) {
            dimensions++;
            type = type.getComponentType();
        }
        return dimensions;
    }

    public static String getDimensions(Object[] arr) {
        String dimensions = arr.length + " ";
        try {
            while (arr.getClass().getComponentType().isArray()) {
                try {
                    arr = (Object[]) arr[0];
                    dimensions += arr.length + " ";
                } catch (ClassCastException f) {
                    try {
                        dimensions += ((float[]) (arr[0])).length;
                    } catch (ClassCastException d) {
                        dimensions += ((double[]) (arr[0])).length;
                    }
                    break;
                } catch (NullPointerException n) {
                    dimensions += "null ";
                }
            }
        } catch (NullPointerException n) {
            dimensions += "null ";
        }
        return dimensions.trim();
    }

    public static Class getDeepType(Object[] arr) {
        Class type = arr.getClass().getComponentType();
        while (type.isArray()) {
            type = type.getComponentType();
        }
        return type;
    }

    public static float[] copy(float[] arr1d) {
        int length = arr1d.length;
        float[] result = new float[length];
        System.arraycopy(arr1d, 0, result, 0, length);
        return result;
    }

    public static float[][] copy(float[][] arr2d) {
        return Arrays.stream(arr2d).map(el -> el.clone()).toArray(a -> arr2d.clone());
    }

    public static float[][][] copy(float[][][] arr3d) {
        return Arrays.stream(arr3d).map(el -> copy(el)).toArray(a -> arr3d.clone());
    }

    public static float[][][][] copy(float[][][][] arr4d) {
        return Arrays.stream(arr4d).map(el -> copy(el)).toArray(a -> arr4d.clone());
    }

    public static float[][][][][] copy(float[][][][][] arr5d) {
        return Arrays.stream(arr5d).map(el -> copy(el)).toArray(a -> arr5d.clone());
    }

    public static Object[] copy(Object[] nDimensionArr) {
        try {
            return Arrays.stream(nDimensionArr).map(el -> copy(Object[].class.cast(el))).toArray(a -> nDimensionArr.clone());
        } catch (ClassCastException e) {
            return Arrays.stream(nDimensionArr).map(el -> copy(float[].class.cast(el))).toArray(a -> nDimensionArr.clone());
        }
    }

    public static float[] flatten(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[(i * cols) + j] = matrix[i][j];
            }
        }
        return result;
    }

    public static float[][] unflatten(float[] flattened, int rows, int cols) {
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = flattened[(i * cols) + j];
            }
        }
        return result;
    }

    public static float[][] convolution(float[][] input, float[][] filter, float bias, int stride) {
        int filterHeight = filter.length;
        int filterWidth = filter[0].length;
        int inputHeight = input.length;
        int inputWidth = input[0].length;
        int resultHeight = (inputHeight - filterHeight) / stride + 1;
        int resultWidth = (inputWidth - filterWidth) / stride + 1;
        float[][] result = new float[resultHeight][resultWidth];
        for (int rRow = 0; rRow < resultHeight; rRow++) {
            int startRow = rRow * stride;
            int boundRow = startRow + filterHeight;
            for (int rCol = 0; rCol < resultWidth; rCol++) {
                int startCol = rCol * stride;
                int boundCol = startCol + filterWidth;
                float val = 0;
                for (int iRow = startRow; iRow < boundRow; iRow++) {
                    int filterRow = iRow - startRow;
                    for (int iCol = startCol; iCol < boundCol; iCol++) {
                        val += input[iRow][iCol] * filter[filterRow][iCol - startCol];
                    }
                }
                result[rRow][rCol] = val + bias;
            }
        }
        return result;
    }

    public static float[][][] convolution(float[][][] input, float[][][][] filters, float[] biases, int stride) {
        int resultDepth = filters.length;
        int filterDepth = filters[0].length;
        int filterHeight = filters[0][0].length;
        int filterWidth = filters[0][0][0].length;
        int resultHeight = (input[0].length - filterHeight) / stride + 1;
        int resultWidth = (input[0][0].length - filterWidth) / stride + 1;
        float[][][] result = new float[resultDepth][resultHeight][resultWidth];
        for (int rDepth = 0; rDepth < resultDepth; rDepth++) {
            float[][][] filter = filters[rDepth];
            for (int rRow = 0; rRow < resultHeight; rRow++) {
                int startRow = rRow * stride;
                int boundRow = startRow + filterHeight;
                for (int rCol = 0; rCol < resultWidth; rCol++) {
                    int startCol = rCol * stride;
                    int boundCol = startCol + filterWidth;
                    float val = 0;
                    for (int iDepth = 0; iDepth < filterDepth; iDepth++) {
                        float[][] filterLayer = filter[iDepth];
                        float[][] inputLayer = input[iDepth];
                        for (int iRow = startRow; iRow < boundRow; iRow++) {
                            int filterRow = iRow - startRow;
                            for (int iCol = startCol; iCol < boundCol; iCol++) {
                                val += inputLayer[iRow][iCol] * filterLayer[filterRow][iCol - startCol];
                            }
                        }
                    }
                    result[rDepth][rRow][rCol] = val + biases[rDepth];
                }
            }
        }
        return result;
    }

    public static float[][] rotate180(float[][] arr2d) {
        int rows = arr2d.length;
        int cols = arr2d[0].length;
        float[][] result = new float[rows][cols];
        int maxrow = rows - 1;
        int maxcol = cols - 1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[maxrow - i][maxcol - j] = arr2d[i][j];
            }
        }
        return result;
    }

    public static float[][] pad(float[][] arr2d, int paddingHeight, int paddingWidth) {
        if (!(paddingHeight < 0 || paddingWidth < 0) || (paddingHeight == 0 && paddingWidth == 0)) {
            int paddedHeight = arr2d.length + paddingHeight + paddingHeight;
            int paddedWidth = arr2d[0].length + paddingWidth + paddingWidth;
            float[][] padded = new float[paddedHeight][paddedWidth];
            int endRow = paddedHeight - paddingHeight;
            int endCol = paddedWidth - paddingWidth;
            for (int j = 0; j < paddingHeight; j++) {//Top
                for (int k = 0; k < paddingWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = endRow; j < paddedHeight; j++) {//Bottom
                for (int k = endCol; k < paddedWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = paddingHeight; j < paddedHeight; j++) {//Left
                for (int k = 0; k < paddingWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = paddingHeight; j < paddedHeight; j++) {//Right
                for (int k = endCol - 1; k < paddingWidth; k++) {
                    padded[j][k] = 0;
                }
            }
            for (int j = paddingHeight; j < endRow; j++) {//Matrix numbers
                for (int k = paddingWidth; k < endCol; k++) {
                    padded[j][k] = arr2d[j - paddingHeight][k - paddingWidth];
                }
            }
            return padded;
        } else {
            return arr2d;
        }
    }

    public static float[][] unpad(float[][] arr2d, int unpaddingHeight, int unpaddingWidth) {
        if (!(unpaddingHeight < 0 || unpaddingWidth < 0) || (unpaddingHeight == 0 || unpaddingWidth == 0)) {
            int resultHeight = arr2d.length - unpaddingHeight - unpaddingHeight;
            int resultWidth = arr2d[0].length - unpaddingWidth - unpaddingWidth;
            float[][] result = new float[resultHeight][resultWidth];
            for (int i = 0; i < resultHeight; i++) {
                for (int j = 0; j < resultWidth; j++) {
                    result[i][j] = arr2d[i + unpaddingHeight][j + unpaddingWidth];
                }
            }
            return result;
        } else {
            return arr2d;
        }
    }

    public static float[][] dilate(float[][] arr2d, int dilationRows, int dilationCols) {
        if (!(dilationRows < 0 || dilationCols < 0) || (dilationRows == 0 || dilationCols == 0)) {
            int rows = arr2d.length;
            int cols = arr2d[0].length;
            float[][] result = new float[rows * (dilationRows + 1) - dilationRows][cols * (dilationCols + 1) - dilationCols];
            int jumpCols = dilationCols + 1;
            int jumpRows = dilationRows + 1;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i * jumpRows][j * jumpCols] = arr2d[i][j];
                }
            }
            return result;
        } else {
            return arr2d;
        }
    }

    public static float[][][] function2dOn3d(float[][][] arr3d, Function<float[][], float[][]> function) {
        int depth = arr3d.length;
        float[][][] result = new float[depth][][];
        for (int i = 0; i < depth; i++) {
            result[i] = function.apply(arr3d[i]);
        }
        return result;
    }

    public static float[][][] bifunction2dOn3d(float[][][] arr3d1, float[][][] arr3d2, BiFunction<float[][], float[][], float[][]> bifunction) {
        int depth = arr3d1.length;
        float[][][] result = new float[depth][][];
        for (int i = 0; i < depth; i++) {
            result[i] = bifunction.apply(arr3d1[i], arr3d2[i]);
        }
        return result;
    }

    public static float[][] functionMatrixVectors(float[][] matrix, Function<float[][], float[][]> operation) {
        int rows = matrix.length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = operation.apply(new float[][]{matrix[i]})[0];
        }
        return result;
    }

    public static float[][] bifunctionMatrixVectors(float[][] matrix, float[] vector, BiFunction<float[][], float[][], float[][]> operation) {
        int rows = matrix.length;
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++) {
            result[i] = operation.apply(new float[][]{matrix[i]}, new float[][]{vector})[0];
        }
        return result;
    }

    public static float[][] function(float[][] matrix, Function<Float, Float> function) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = function.apply(matrix[i][j]);
            }
        }
        return result;
    }

    public static float[][] bifunction(float[][] m1, float[][] m2, BiFunction<Float, Float, Float> bifunction) {
        int rows = m1.length;
        int columns = m1[0].length;
        float[][] result = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result[i][j] = bifunction.apply(m1[i][j], m2[i][j]);
            }
        }
        return result;
    }
}
