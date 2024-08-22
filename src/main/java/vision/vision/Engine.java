package vision.vision;

import java.util.LinkedList;
import java.util.Random;

import vision.matrix.Matrix;

public class Engine {
	private LinkedList<Transform> transforms= new LinkedList<>();
	private LinkedList<Matrix> weights = new LinkedList<>();
	private LinkedList<Matrix> biases = new LinkedList<>();
	
	private LossFunction lossFunction = LossFunction.CROSSENTROPY;
	private boolean storeInputError = false;
	private double scaleInitialWeights = 1;
	
	BatchResult runForwards(Matrix input) {
		BatchResult batchResult = new BatchResult();
		Matrix output = input;
		
		int denseIndex = 0;
		
		batchResult.addIo(output);
		
		for(var t: transforms) {
			if(t == Transform.DENSE) {
				
				batchResult.addWeightInput(output);
				Matrix weight = weights.get(denseIndex);
				Matrix bias = biases.get(denseIndex);
				
				output = weight.multiply(output).modify((row, col, value) -> value + bias.get(row));
						
				++denseIndex;
			}
			else if(t == Transform.RELU) {
				output = output.modify(value -> value > 0 ? value: 0);
			}
			else if(t == Transform.SOFTMAX) {
				output = output.softMax();
			}
			
			batchResult.addIo(output);
		}
		
		return batchResult;
	}
	
	public void runBackwards(BatchResult batchResult, Matrix expected) {
		var transformsIt = transforms.descendingIterator();
		
		if(lossFunction != LossFunction.CROSSENTROPY || transforms.getLast() != Transform.SOFTMAX) {
			throw new UnsupportedOperationException("Loss function must be cross entropy and last transform must be softmax");
		}
		
		var ioIt = batchResult.getIo().descendingIterator();
		var weightIt = weights.descendingIterator();
		
		Matrix softmaxOutput = ioIt.next();
		Matrix error = softmaxOutput.apply((index, value)->value - expected.get(index));
		
		while(transformsIt.hasNext()) {
			Transform transform = transformsIt.next();
			Matrix input = ioIt.next();
			
			switch(transform) {
			case DENSE:
				Matrix weight = weightIt.next();
				
				batchResult.addWeightError(error);
				
				if(weightIt.hasNext() || storeInputError) {
					error = weight.transpose().multiply(error);
				}
				break;
			case RELU:
				error = error.apply((index, value)->input.get(index) > 0 ? value: 0);
				break;
			case SOFTMAX:
				break;
			default:
				throw new UnsupportedOperationException("Not implemented");
			}
			
			//System.out.println(transform);
		}
		
		if(storeInputError) {
			batchResult.setInputError(error);
		}
	}

	
	public void setStoreInputError(boolean storeInputError) {
		this.storeInputError = storeInputError;
	}

	public void add(Transform transform, double... params) {
		Random random = new Random();
		
		if(transform == Transform.DENSE) {
			int numberNeurons = (int)params[0];
			int weightsPerNeuron = weights.size() == 0 ? (int)params[1]: weights.getLast().getRows();
			
			Matrix weight = new Matrix(numberNeurons, weightsPerNeuron, i->scaleInitialWeights * random.nextGaussian());
			Matrix bias = new Matrix(numberNeurons, 1, i->0);
			
			weights.add(weight);
			biases.add(bias);
		}
		transforms.add(transform);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		sb.append(String.format("Scale initial weights: %.3f\n", scaleInitialWeights));
		sb.append("\nTransforms:\n");

		int weightIndex = 0;
		for (var t : transforms) {
			
			sb.append(t);
			
			if(t == Transform.DENSE) {
				sb.append(" ").append(weights.get(weightIndex).toString(false));
				
				weightIndex++;
			}
			
			sb.append("\n");
		}

		return sb.toString();
	}
}
