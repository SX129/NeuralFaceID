package vision;

import java.io.File;

import vision.neuralnetwork.NeuralNetwork;
import vision.neuralnetwork.Transform;
import vision.neuralnetwork.loader.Loader;
import vision.neuralnetwork.loader.MetaData;
import vision.neuralnetwork.loader.image.ImageLoader;

public class App {

	public static void main(String[] args) {
		final String filename = "mnistNeural0.net";

		if (args.length == 0 || !new File(args[0]).isDirectory()) {
			System.out.println("Usage: [app] <MNIST DATA DIRECTORY>");
			return;
		}

		String directory = args[0];

		final String trainImages = String.format("%s%s%s", directory, File.separator, "train-images.idx3-ubyte");
		final String trainLabels = String.format("%s%s%s", directory, File.separator, "train-labels.idx1-ubyte");
		final String testImages = String.format("%s%s%s", directory, File.separator, "t10k-images.idx3-ubyte");
		final String testLabels = String.format("%s%s%s", directory, File.separator, "t10k-labels.idx1-ubyte");

		Loader trainLoader = new ImageLoader(trainImages, trainLabels, 32);
		Loader testLoader = new ImageLoader(testImages, testLabels, 32);

		System.out.println(Runtime.getRuntime().availableProcessors() + " processors available");

		MetaData metaData = trainLoader.open();
		int inputSize = metaData.getInputSize();
		int outputSize = metaData.getExpectedSize();
		trainLoader.close();

		NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);

		if (neuralNetwork == null) {
			System.out.println("Unable to load neural network from saved file. Creating new network.");

			neuralNetwork = new NeuralNetwork();
			
			neuralNetwork.setScaleInitialWeights(0.2);
			neuralNetwork.setThreads(32);
			neuralNetwork.setEpochs(100);
			neuralNetwork.setLearningRate(0.02, 0.001);
			
			neuralNetwork.add(Transform.DENSE, 200, inputSize);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, outputSize);
			neuralNetwork.add(Transform.SOFTMAX);

			
		} else {
			System.out.println("Neural network loaded from saved file: " + filename + "\n");
		}

		System.out.println(neuralNetwork);

		neuralNetwork.fit(trainLoader, testLoader);
		if (neuralNetwork.save(filename)) {
			System.out.println("Neural network saved to file: " + filename);
		} else {
			System.out.println("Unable to save neural network to: " + filename);
		}
	}

}
