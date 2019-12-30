package neuralnetwork;
import static neuralnetwork.NN.*;
public class Layer {
    double[][] weights;
    double[][] biases;
    Layer(int previousNodes, int nodes){
        weights = create(previousNodes,nodes,0);
        biases = create(1,nodes,0);
        weights = scale( randomize(weights,2,-1), Math.sqrt(2.0/previousNodes) );
        biases = randomize(biases,2,-1);
    }
}
