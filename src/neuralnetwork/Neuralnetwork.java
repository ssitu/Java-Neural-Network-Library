package neuralnetwork;
public class Neuralnetwork {
    public static void main(String[] args) {
        NN nn = new NN("tanh","softmax",2,4,3,2,1,2,3,4,3);
        double[][] inputs = new double[1][2];
        inputs[0][0] = 1;
        inputs[0][1] = -9;
        double[][] targets = new double[1][3];
        targets[0][0] = 0;
        targets[0][1] = 0;
        targets[0][2] = 1;
        while(true){
            NN.print(nn.feedforward(inputs), "feedforward");
            nn.backpropagation(inputs, targets);
        }
    }
}
