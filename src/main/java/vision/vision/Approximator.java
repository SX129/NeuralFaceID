package vision.vision;

import java.util.function.Function;

import vision.matrix.Matrix;

public class Approximator {
	
	public static Matrix gradient(Matrix input, Function<Matrix, Matrix> transform) {
		final double INC = 0.00000001;
		
		Matrix loss1 = transform.apply(input);
		
		assert loss1.getCols() == input.getCols(): "Input and loss columns are not equal";
		assert loss1.getRows() == 1: "Transform does not return a row vector";
		
		Matrix result = new Matrix(input.getRows(), input.getCols(), i -> 0);

		input.forEach((row, col, index, value) -> {
			Matrix incremented = input.addIncrement(row, col, INC);
			Matrix loss2 = transform.apply(incremented);
			
			double rate = (loss2.get(col) - loss1.get(col)) / INC;
			
			result.set(row, col, rate);
		});
		
		return result;
	}
}
