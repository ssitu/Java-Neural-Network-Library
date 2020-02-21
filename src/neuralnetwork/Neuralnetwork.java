package neuralnetwork;

import java.util.ArrayList;

public class Neuralnetwork {
    public static void main(String[] args) {
        class Data{
            float[][] inputs;
            float[][] targets;
            void addInputs(float ... inputs){
                this.inputs = new float[1][inputs.length];
                this.inputs[0] = inputs;
            }
            void addTargets(float ... targets){
                this.targets = new float[1][targets.length];
                this.targets[0] = targets;
            }
        }
        NNest.NN nn = new NNest().new NN(.001,"leakyrelu","sigmoid","quadratic","momentum",false,2,800,800,1);
        System.out.println(nn.NETWORKSIZE);
        System.out.println(nn.toString());
        ArrayList<Data> data = new ArrayList<>();
        data.add(new Data());
        data.add(new Data());
        data.add(new Data());
        data.add(new Data());
        data.get(0).addInputs(1,1);
        data.get(0).addTargets(0);
        data.get(1).addInputs(0,1);
        data.get(1).addTargets(1);
        data.get(2).addInputs(1,0);
        data.get(2).addTargets(1);
        data.get(3).addInputs(0,0);
        data.get(3).addTargets(0);
        NNest.startGraph();
        for(int i = 0; i < 250000; i--){
            int random = (int)(Math.random()*4);
//                nn.print(data.get(random).inputs, "inputs");
//                nn.print(nn.feedforward(data.get(random).inputs), "feedforward");
            nn.backpropagation(data.get(random).inputs, data.get(random).targets);
        }
        
    }
}
