package vision;

import vision.neuralnetwork.NeuralNetwork;
import vision.neuralnetwork.Transform;
import vision.neuralnetwork.loader.Loader;
import vision.neuralnetwork.loader.test.TestLoader;

public class GeneratedDataApp {
	public static void main(String[] args) {
		String filename = "neural1.net";
		System.out.println(Runtime.getRuntime().availableProcessors() + " processors available");
		
		NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);
		
		if (neuralNetwork == null) {
			System.out.println("Unable to load neural network from saved file. Creating new network.");
			
			int inputRows = 10;
			int outputRows = 3;
			
			neuralNetwork = new NeuralNetwork();
			
			neuralNetwork.add(Transform.DENSE, 100, inputRows);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, 50);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, outputRows);
			neuralNetwork.add(Transform.SOFTMAX);
			
			neuralNetwork.setThreads(32);
			neuralNetwork.setEpochs(100);
			neuralNetwork.setLearningRate(0.02, 0.001);
		}else {
			System.out.println("Neural network loaded from saved file: " + filename + "\n");
		}
		
		System.out.println(neuralNetwork);

		Loader trainLoader = new TestLoader(60_000, 32);
		Loader testLoader = new TestLoader(10_000, 32);
		
		neuralNetwork.fit(trainLoader, testLoader);
		if(neuralNetwork.save(filename)) {
			System.out.println("Neural network saved to file: " + filename);
		}else {
			System.out.println("Unable to save neural network to: " + filename);
		}
	}
}
