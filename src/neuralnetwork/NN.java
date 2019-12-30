package neuralnetwork;
import java.util.*;
import java.util.function.BiFunction;
public class NN {
    ArrayList<Layer> network = new ArrayList<>();
    final double STARTINGLR = .001;
    double lr;
    final int NETWORKSIZE;
    BiFunction<double[][],Boolean,double[][]> activationHiddens;
    BiFunction<double[][],Boolean,double[][]> activationOutputs;
    NN(String hiddenActivationFunction, String outputActivationFunction, int ... layerNodes){//Randomize weights and biases
        if(layerNodes.length < 3){
            throw new IllegalArgumentException("MUST HAVE MORE THAN 2 LAYERS IN THE NEURAL NETWORK");
        }
        lr = STARTINGLR;
        //Activation functions for the hidden layers
        if("sigmoid".equalsIgnoreCase(hiddenActivationFunction)){
            activationHiddens = (x,y) -> sigmoidActivation(x,y);
        }
        else if("tanh".equalsIgnoreCase(hiddenActivationFunction)){
            activationHiddens = (x,y) -> tanhActivation(x,y);
        }
        else if("leakyrelu".equalsIgnoreCase(hiddenActivationFunction)){
            activationHiddens = (x,y) -> leakyReluActivation(x,y);
        }
        else{
            throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE HIDDEN LAYERS");
        }
        //Activation functions for the output layer
        if("sigmoid".equalsIgnoreCase(outputActivationFunction)){
            activationOutputs = (x,y) -> sigmoidActivation(x,y);
        }
        else if("tanh".equalsIgnoreCase(outputActivationFunction)){
            activationOutputs = (x,y) -> tanhActivation(x,y);
        }
        else if("softmax".equalsIgnoreCase(outputActivationFunction)){
            activationOutputs = (x,y) -> softmaxActivation(x,y);
        }
        else{
            throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE OUTPUT LAYER");
        }
        //Adding each layer
        for(int i = 1; i < layerNodes.length; i++){
            Layer layer = new Layer(layerNodes[i-1],layerNodes[i]);
            network.add(layer);
        }
        NETWORKSIZE = network.size();
    }
    double[][] feedforward(double[][] inputs){
        double[][] outputs = inputs;
        for(int i = 0; i < NETWORKSIZE-1; i++){//Feed the inputs through the hidden layers
            outputs = activationHiddens.apply(add(dot(outputs,network.get(i).weights),network.get(i).biases),false);
        }
        //Feed the output from the hidden layers to the output layers with its (different) activation function
        outputs = activationOutputs.apply(add(dot(outputs,network.get(NETWORKSIZE-1).weights),network.get(NETWORKSIZE-1).biases),false);
        return outputs;
    }
    void backpropagation(double[][] inputs, double[][] targets){//Using notation from neuralnetworksanddeeplearning.com
        double[][] outputs = inputs;
        //Each partial derivative is used in this order
        double[][] dC_dA;
        double[][] dA_dZ;
        double[][] dZ_dW;
        double[][] dC_dZ = create(0,0,0);
        double[][] dC_dW;
        double[][] bGradients;
        double[][] wGradients;
        double[][] dZ_dA = create(0,0,0);
        ArrayList<double[][]> Z = new ArrayList<>();//"Z" = the unactivated inputs from the weights and biases
        ArrayList<double[][]> A = new ArrayList<>();//"A" = the activated "Z"s
        A.add(outputs);
        for(int i = 0; i < NETWORKSIZE-1; i++){
            outputs = add(dot(outputs,network.get(i).weights),network.get(i).biases);//Computing "Z"
            Z.add(outputs);
            outputs = activationHiddens.apply(outputs, false);//Computing "A"
            A.add(outputs);
        }
        outputs = add(dot(outputs,network.get(NETWORKSIZE-1).weights),network.get(NETWORKSIZE-1).biases);
        Z.add(outputs);
        outputs = activationOutputs.apply(outputs, false);
        A.add(outputs);
        //Using the Mean Squared Error cost function to compute its gradient easily
        dC_dA = subtract(A.get(NETWORKSIZE), targets);
        boolean outputLayer = true;
        for(int i = NETWORKSIZE; i > 0; i--){
            if(!outputLayer){
                dZ_dA = network.get(i).weights;
            }
            if(outputLayer){
                dA_dZ = activationOutputs.apply(Z.get(i-1),true);
            }
            else{
                dA_dZ = activationHiddens.apply(Z.get(i-1),true);
            }
            dZ_dW = A.get(i-1);
            if(!outputLayer){
                dC_dA = dot(dC_dZ,transpose(dZ_dA));
            }
            if(outputLayer){
                dC_dZ = multiply(dA_dZ,dC_dA);
            }
            else{
                dC_dZ = multiply(dC_dA,dA_dZ);
            }
            dC_dW = dot(transpose(dZ_dW),dC_dZ);
            bGradients = scale(lr,dC_dZ);
            wGradients = scale(lr,dC_dW);
            network.get(i-1).biases = subtract(network.get(i-1).biases,bGradients);
            network.get(i-1).weights = subtract(network.get(i-1).weights,wGradients);
            if(outputLayer){
                outputLayer = false;
            }
        }
    }
    private double[][] softmaxActivation(double[][] matrix, boolean derivative){
        int rows = matrix.length;
        int columns = matrix[0].length;
        double[][] matrixResult = new double[rows][columns];
        double sum = 0;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                sum += Math.exp(matrix[i][j]);
            }
        }
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                matrixResult[i][j] = Math.exp(matrix[i][j])/sum;
            }
        }
        if(!derivative){
            return matrixResult;
        }
        double[][] derivatives = new double[rows][columns];
        double[][] ones = create(rows,columns,1);
        derivatives = multiply(matrixResult,subtract(ones,matrixResult));
        return derivatives;
    }
    private double leakyRelu(double x, boolean derivative){
        if(derivative == true){
            if(x < 0)
                return .001;
            else
                return 1;
        }
        else{
            if(x < 0)
                return .001*x;
            else
                return x;
        }
    }
    private double[][] leakyReluActivation(double[][] matrix, boolean derivative){
        double[][] matrixResult = new double[matrix.length][matrix[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = leakyRelu(matrix[i][j], derivative);
        return matrixResult;
    }
    private double sigmoid(double x, boolean derivative){
        if(derivative == true)
            return sigmoid(x, false)*(1 - sigmoid(x, false));//sigmoid'(x)
        return 1/(1+Math.exp(-x));//sigmoid(x)
    }
    private double[][] sigmoidActivation(double[][] matrix, boolean derivative){
            double[][] matrixResult = new double[matrix.length][matrix[0].length];
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = sigmoid(matrix[i][j], derivative);
            return matrixResult;
    }
    private double tanh(double x, boolean derivative){
        if(derivative == true)
            return 1 - (Math.pow(tanh(x, false), 2));//tanh'(x)
        return (2/(1 + Math.exp(-2*x)))-1;//tanh(x)
    }
    private double[][] tanhActivation(double[][] matrix, boolean derivative){
            double[][] matrixResult = new double[matrix.length][matrix[0].length];
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = tanh(matrix[i][j], derivative);
            return matrixResult;
    }
    
    
    
    
    static private void sizeException(double[][] matrix){
        for(double[] matrixRow : matrix) 
            if(matrixRow.length != matrix[0].length) 
                throw new IllegalArgumentException("Inconsistent Matrix Size");  
    }
    static private void multiplicationDimensionMismatch(double[][] matrixA, double[][] matrixB){
        if(matrixA[0].length != matrixB.length)//A columns must equal B rows
            throw new IllegalArgumentException("Matrices Dimension Mismatch");
    }
    static private void dimensionMismatch(double[][] matrixA, double[][] matrixB){
        if(matrixA.length != matrixB.length || matrixA[0].length != matrixB[0].length)
            throw new IllegalArgumentException("Matrices Dimension Mismatch");
    }
    static void print(double[][] matrix, String nameOfMatrix){
        System.out.println(nameOfMatrix + ": ");
        int rows = matrix.length;
        int columns = matrix[0].length;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++)
                System.out.print("[" + matrix[i][j] + "] ");
            System.out.println("");
        }
    }
    static void resetDouble(double[][] matrix){
        int rows = matrix.length;
        int columns = matrix[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrix[i][j] = 0;
    }
    static void resetInt(int[][] matrix){
        int rows = matrix.length;
        int columns = matrix[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrix[i][j] = 0;
    }
    static double[][] create(int rows, int columns, double valueToAllElements){
        double[][] matrixResult = new double[rows][columns];
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++){
                matrixResult[i][j] = valueToAllElements;
            }
        return matrixResult;
    }
    static double[][] randomize(double[][] matrix, double range, double minimum){
        sizeException(matrix);
        double[][] matrixResult = matrix;
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = Math.random() * range + minimum;
        return matrixResult;
    }
    static double[][] transpose(double[][] matrix){
        sizeException(matrix);
        double[][] matrixResult = new double[matrix[0].length][matrix.length];
        int rows = matrix.length;
        int columns = matrix[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[j][i] = matrix[i][j];
        return matrixResult;
    }
    static double[][] scale(double[][] matrix, double factor){
        sizeException(matrix);
        double[][] matrixResult = new double[matrix.length][matrix[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = factor * matrix[i][j];
        return matrixResult;   
    }
    static double[][] scale(double factor, double[][] matrix){
        sizeException(matrix);
        double[][] matrixResult = new double[matrix.length][matrix[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = factor * matrix[i][j];
        return matrixResult;   
    }
    static double[][] add(double[][] matrixA, double[][] matrixB){
        sizeException(matrixA);
        sizeException(matrixB);
        dimensionMismatch(matrixA, matrixB);
        double[][] matrixResult = new double[matrixA.length][matrixA[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = matrixA[i][j] + matrixB[i][j];
        return matrixResult;
    }
    static double[][] subtract(double[][] matrixA, double[][] matrixB){
        sizeException(matrixA);
        sizeException(matrixB);
        dimensionMismatch(matrixA, matrixB);
        double[][] matrixResult = new double[matrixA.length][matrixA[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = matrixA[i][j] - matrixB[i][j];
        return matrixResult;
    }
    static double[][] multiply(double[][] matrixA, double[][] matrixB){
        sizeException(matrixA);
        sizeException(matrixB);
        dimensionMismatch(matrixA, matrixB);
        double[][] matrixResult = new double[matrixA.length][matrixA[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = matrixA[i][j] * matrixB[i][j];
        return matrixResult;
    }
    static double[][] dot(double[][] matrixA, double[][] matrixB){
        //make sure lengths of rows are the same for each matrix
        sizeException(matrixA);
        sizeException(matrixB);
        multiplicationDimensionMismatch(matrixA, matrixB);
        double[][] matrixResult = new double[matrixA.length][matrixB[0].length];//result matrix
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        //multiply A row elements by B column elements = 1 element in result matrix
        for(int i = 0; i < rows; i++)//A rows or B columns or Result rows
            for(int j = 0; j < columns; j++)//A rows or B columns or Result columns
                for(int k = 0; k < matrixA[0].length; k++)//A columns or B rows
                    matrixResult[i][j] += matrixA[i][k] * matrixB[k][j];
        return matrixResult;
    }
    static double[][] power(double[][] matrix, double power){
        double[][] matrixResult = new double[matrix.length][matrix[0].length];
        int rows = matrixResult.length;
        int columns = matrixResult[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                matrixResult[i][j] = Math.pow(matrix[i][j], power);
        return matrixResult;
    }
    static double sum(double[][] matrix){
        double sum = 0;
        int rows = matrix.length;
        int columns = matrix[0].length;
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                sum += matrix[i][j];
        return sum;
    }
}
