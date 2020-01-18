package neuralnetwork;
import java.io.Serializable;
import java.util.*;
import java.util.function.*;
import javafx.application.*;
import static javafx.application.Application.launch;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.stage.Stage;
import java.io.*;
public class NNest extends Application implements Serializable{
    volatile static double globalCost;
    volatile static int increment = 0;
    public class NN implements Serializable{
        private class Layer implements Serializable{
            double[][] weights;
            double[][] biases;
            Layer(int previousNodes,int nodes){
                weights = create(previousNodes,nodes,0);
                biases = create(1,nodes,0);
                weights = scale(randomize(weights,2,-1),Math.sqrt(2.0/previousNodes));
                biases = randomize(biases,2,-1);
            }
        }
        ArrayList<Layer> network = new ArrayList<>();
        double lr;
        double cost;
        final int NETWORKSIZE;//Total layers not including the input layer
        transient BiFunction<double[][],Boolean,double[][]> activationHiddens;
        transient BiFunction<double[][],Boolean,double[][]> activationOutputs;
        transient BiFunction<double[][], double[][], Function<Boolean, double[][]>> costFunction;
        NN(double learningRate, String hiddenActivationFunction, String outputActivationFunction, String costFunction, int ... layerNodes){
            lr = learningRate;
            if(layerNodes.length < 3){
                throw new IllegalArgumentException("MUST HAVE MORE THAN 2 LAYERS IN THE NEURAL NETWORK");
            }
            //Activation functions for the hidden layers
            if("sigmoid".equalsIgnoreCase(hiddenActivationFunction)){
                activationHiddens = (x,y) -> sigmoidActivation(x,y);
            }
            else if("tanh".equalsIgnoreCase(hiddenActivationFunction)){
                activationHiddens = (x,y) -> tanhActivation(x,y);
            }
            else if("relu".equalsIgnoreCase(hiddenActivationFunction)){
                activationHiddens = (x,y) -> reluActivation(x,y);
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
            else if("linear".equalsIgnoreCase(outputActivationFunction)){
                activationOutputs = (x,y) -> linearActivation(x,y);
            }
            else{
                throw new IllegalArgumentException("INVALID ACTIVATION FUNCTION FOR THE OUTPUT LAYER");
            }
            //Cost functions for the backpropagation
            if("quadratic".equalsIgnoreCase(costFunction)){
                this.costFunction = (x,y) -> (z) -> quadraticLoss(x,y,z);
            }
            else if("log".equalsIgnoreCase(costFunction)){//The target should be passed as making the correct classification as the highest value
                this.costFunction = (x,y) -> (z) -> logLoss(x,y,z);
            }
            else{
                throw new IllegalArgumentException("INVALID COST FUNCTION");
            }
            //Adding each layer
            for(int i = 1; i < layerNodes.length; i++){
                Layer layer = new Layer(layerNodes[i-1],layerNodes[i]);
                network.add(layer);
            }
            NETWORKSIZE = network.size();
        }
        public String getNetworkSize(){
            String networkLayers = "";
            for(Layer layer : network){
                networkLayers += layer.weights.length + ",";
            }
            networkLayers += network.get(NETWORKSIZE-1).weights[0].length;
            return networkLayers;
        }
        public void save(){
        try{
            FileOutputStream fileOut = new FileOutputStream(System.getProperty("user.dir") + "/neuralnetwork(" + this.getNetworkSize() + ").ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this.network);
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }
    public void load(){
        try{
            FileInputStream fileIn = new FileInputStream(System.getProperty("user.dir") + "/neuralnetwork(" + this.getNetworkSize() + ").ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            this.network = (ArrayList)in.readObject();
        }
        catch (IOException | ClassNotFoundException e) {
            System.out.println("Could not load network settings");
        }
    }
        public double[][] feedforward(double[][] inputs){
            double[][] outputs = inputs;
            for(int i = 0; i < NETWORKSIZE-1; i++){//Feed the inputs through the hidden layers
                outputs = activationHiddens.apply(add(dot(outputs,network.get(i).weights),network.get(i).biases),false);
            }
            //Feed the output from the hidden layers to the output layers with its (different) activation function
            outputs = activationOutputs.apply(add(dot(outputs,network.get(NETWORKSIZE-1).weights),network.get(NETWORKSIZE-1).biases),false);
            return outputs;
        }
        public void backpropagation(double[][] inputs, double[][] targets){//Using notation from neuralnetworksanddeeplearning.com
            if(targets[0].length != network.get(NETWORKSIZE-1).biases[0].length){
                throw new IllegalArgumentException("TARGETS ARRAY DO NOT MATCH THE SIZE OF THE OUTPUT LAYER");
            }
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
            dC_dA = costFunction.apply(copy(A.get(NETWORKSIZE)), targets).apply(true);
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
                costFunction.apply(copy(A.get(NETWORKSIZE)), targets).apply(false);//Update the cost to anneal the learning rate
                //Square Root Annealing
//                bGradients = scale(lr*Math.sqrt(cost)*Math.pow(10, -Math.pow(NETWORKSIZE,1/NETWORKSIZE)),dC_dZ);
//                wGradients = scale(lr*Math.sqrt(cost)*Math.pow(10, -Math.pow(NETWORKSIZE,1/NETWORKSIZE)),dC_dW);
//                double stabilizer = lr*Math.sqrt(cost)*Math.pow(10, -Math.pow(NETWORKSIZE, .8*Math.pow(NETWORKSIZE, -1)));
//                double stabilizer = lr*Math.sqrt(cost) * Math.pow(10, -NETWORKSIZE);
//                double stabilizer = lr*Math.sqrt(cost)* Math.pow(10, -NETWORKSIZE+2+(NETWORKSIZE*Math.E*(.02209*NETWORKSIZE-.0182)));
//                double stabilizer = lr*Math.sqrt(cost)* Math.pow(10, -NETWORKSIZE+2+(NETWORKSIZE*Math.E*(.02209*NETWORKSIZE-.02)));
                double stabilizer = lr*Math.sqrt(cost)* Math.pow(10, -NETWORKSIZE+2+(NETWORKSIZE*Math.E*((.02+.0002*NETWORKSIZE)*NETWORKSIZE-(.02+.0002*NETWORKSIZE))));
                bGradients = scale(stabilizer,dC_dZ);
                wGradients = scale(stabilizer,dC_dW);
//                bGradients = scale(lr*cost,dC_dZ);
//                wGradients = scale(lr*cost,dC_dW);
                network.get(i-1).biases = subtract(network.get(i-1).biases,bGradients);
                network.get(i-1).weights = subtract(network.get(i-1).weights,wGradients);
                if(outputLayer){
                    outputLayer = false;
                }
            }
            increment++;
            globalCost = cost;
        }
        private double[][] linearActivation(double[][] matrix, boolean derivative){
            int rows = matrix.length;
            int columns = matrix[0].length;
            if(!derivative){
                return matrix;
            }
            else{
                double[][] matrixResult = create(rows, columns, 1);
                return matrixResult;
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
            double[][] ones = create(rows,columns,1);
            double[][] derivatives = multiply(matrixResult,subtract(ones,matrixResult));
            return derivatives;
        }
        private double relu(double x, boolean derivative){
            if(derivative == true){
                if(x < 0)
                    return 0;
                else
                    return 1;
            }
            else{
                if(x < 0)
                    return 0;
                else
                    return x;
            }
        }
        private double[][] reluActivation(double[][] matrix, boolean derivative){
            double[][] matrixResult = new double[matrix.length][matrix[0].length];
            int rows = matrixResult.length;
            int columns = matrixResult[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = relu(matrix[i][j], derivative);
            return matrixResult;
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
        private double[][] quadraticLoss(double[][] outputs, double[][] targets, boolean derivative){
            if(derivative){
                return subtract(outputs, targets);
            }
            else{
                double[][] loss = scale(power(subtract(outputs, targets), 2), .5);
                int rows = loss.length;
                int columns = loss[0].length;
                double total = 0;
                for(int i = 0; i < rows; i++)
                    for(int j = 0; j < columns; j++){
                        total += loss[i][j];
                    }
                cost = total/columns;
                return null;
            }
        }
        private double[][] logLoss(double[][] outputs, double[][] targets, boolean derivative){
            int columns = targets[0].length;//Should be the same as outputs[0].length
            double[] temp;
            temp = targets[0].clone();
            Arrays.sort(temp);
            try{
                if(temp[columns-1] == temp[columns-2]){
                    throw new IllegalArgumentException("INDICATE CORRECT CLASSIFICATION WITH A HIGHEST VALUE IN THE TARGET ARRAY");
                }
            }
            catch(java.lang.ArrayIndexOutOfBoundsException e){
                throw new IllegalArgumentException("AT LEAST TWO TARGETS NEEDED FOR LOG LOSS");
            }
            double max = Double.NEGATIVE_INFINITY;
            int correctClass = 0;
            for(int i = 0; i < columns; i++){
                if(targets[0][i] > max){
                    max = targets[0][i];
                    correctClass = i;//Note the correct classification
                }
            }//Found the correct classification index
            if(derivative){
                for(int i = 0; i < columns; i++){
                    if(i == correctClass){
                        outputs[0][i] -= 1;//The derivative of the -log of the correct classification is the predicted probability - 1
                    }
                }
                return outputs;//Otherwise the derivatives of the wrong classifications are the outputs
            }
            else{
                cost = -Math.log(outputs[0][correctClass]);
                return null;
            }
        }





        private void sizeException(double[][] matrix){
            for(double[] matrixRow : matrix) 
                if(matrixRow.length != matrix[0].length) 
                    throw new IllegalArgumentException("Inconsistent Matrix Size");  
        }
        private void multiplicationDimensionMismatch(double[][] matrixA, double[][] matrixB){
            if(matrixA[0].length != matrixB.length)//A columns must equal B rows
                throw new IllegalArgumentException("Matrices Dimension Mismatch");
        }
        private void dimensionMismatch(double[][] matrixA, double[][] matrixB){
            if(matrixA.length != matrixB.length || matrixA[0].length != matrixB[0].length)
                throw new IllegalArgumentException("Matrices Dimension Mismatch");
        }
        public void print(double[][] matrix, String nameOfMatrix){
            System.out.println(nameOfMatrix + ": ");
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < columns; j++)
                    System.out.print("[" + matrix[i][j] + "] ");
                System.out.println("");
            }
        }
        public void resetDouble(double[][] matrix){
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrix[i][j] = 0;
        }
        public void resetInt(int[][] matrix){
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrix[i][j] = 0;
        }
        public double[][] create(int rows, int columns, double valueToAllElements){
            double[][] matrixResult = new double[rows][columns];
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++){
                    matrixResult[i][j] = valueToAllElements;
                }
            return matrixResult;
        }
        public double[][] randomize(double[][] matrix, double range, double minimum){
            sizeException(matrix);
            double[][] matrixResult = matrix;
            int rows = matrixResult.length;
            int columns = matrixResult[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = Math.random() * range + minimum;
            return matrixResult;
        }
        public double[][] transpose(double[][] matrix){
            sizeException(matrix);
            double[][] matrixResult = new double[matrix[0].length][matrix.length];
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[j][i] = matrix[i][j];
            return matrixResult;
        }
        public double[][] scale(double[][] matrix, double factor){
            sizeException(matrix);
            double[][] matrixResult = new double[matrix.length][matrix[0].length];
            int rows = matrixResult.length;
            int columns = matrixResult[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = factor * matrix[i][j];
            return matrixResult;   
        }
        public double[][] scale(double factor, double[][] matrix){
            sizeException(matrix);
            double[][] matrixResult = new double[matrix.length][matrix[0].length];
            int rows = matrixResult.length;
            int columns = matrixResult[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = factor * matrix[i][j];
            return matrixResult;   
        }
        public double[][] add(double[][] matrixA, double[][] matrixB){
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
        public double[][] subtract(double[][] matrixA, double[][] matrixB){
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
        public double[][] multiply(double[][] matrixA, double[][] matrixB){
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
        public double[][] dot(double[][] matrixA, double[][] matrixB){
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
        public double[][] power(double[][] matrix, double power){
            double[][] matrixResult = new double[matrix.length][matrix[0].length];
            int rows = matrixResult.length;
            int columns = matrixResult[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    matrixResult[i][j] = Math.pow(matrix[i][j], power);
            return matrixResult;
        }
        public double sum(double[][] matrix){
            double sum = 0;
            int rows = matrix.length;
            int columns = matrix[0].length;
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    sum += matrix[i][j];
            return sum;
        }
        public double[][] copy(double[][] matrix){
            int rows = matrix.length;
            int columns = matrix[0].length;
            double[][] matrixResult = new double[rows][columns];
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < columns; j++){
                    matrixResult[i][j] = matrix[i][j];
                }
            }
            return matrixResult;
        }
    }
    @Override
    public void start(Stage stage){
        boolean costVSAccuracy = false; //Cost = true, Accuracy = false
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setAnimated(false);
        xAxis.setLabel("Training Sessions");
        yAxis.setAnimated(false);
        yAxis.setLabel(costVSAccuracy ? "Cost" : "Accuracy"); 
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(yAxis.getLabel() + " over " + xAxis.getLabel());
        ScatterChart<Number, Number> chart = new ScatterChart<>(xAxis, yAxis);
        chart.setAnimated(false);
        chart.getData().add(series);
        Scene scene = new Scene(chart, 600, 300);
        stage.setScene(scene);
        stage.show();
        Thread updateThread = new Thread(() -> {
            while(true){
                try{
                    Thread.sleep(50);
                    Platform.runLater(() -> series.getData().add(new XYChart.Data<>(increment, costVSAccuracy ? globalCost : 1/Math.pow(10, globalCost))));
                } 
                catch(InterruptedException e){
                    throw new RuntimeException(e);
                }
            }
        });
        updateThread.setDaemon(true);
        updateThread.start();
    }
    public static void main(String[] args){
        launch(args);
    }
}