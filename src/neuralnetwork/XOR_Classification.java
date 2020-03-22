package neuralnetwork;

import java.util.ArrayList;

public class XOR_Classification {

    public static void main(String[] args) {
        class Data {

            float[][] inputs;
            float[][] targets;

            Data(float[] inputs, float[] targets) {
                this.inputs = new float[][]{inputs};
                this.targets = new float[][]{targets};
            }
        }
        NNest.NN nn = new NNest().new NN("xor_class", .001, 7777, "sigmoid", "softmax", "log", "nadam", 2, 2, 2);
        System.out.println(nn.NETWORKSIZE);
        System.out.println(nn.toString());
        ArrayList<Data> data = new ArrayList<>();
        //First output = true, second output = false;
        data.add(new Data(new float[]{1, 1}, new float[]{0, 1}));
        data.add(new Data(new float[]{0, 1}, new float[]{1, 0}));
        data.add(new Data(new float[]{1, 0}, new float[]{1, 0}));
        data.add(new Data(new float[]{0, 0}, new float[]{0, 1}));
        NNest.graph(false);
        for (int i = 0; i < 10000000; i++) {
            int random = (int) (Math.random() * 4);
                nn.print(data.get(random).inputs, "inputs");
                nn.print(nn.feedforward(data.get(random).inputs), "feedforward");
            nn.backpropagation(data.get(random).inputs, data.get(random).targets);
        }
        System.exit(0);
    }
}
