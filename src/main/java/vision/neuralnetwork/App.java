package vision.neuralnetwork;

//@formater:on
public class App
{
	// perceptron neuron
	static double neuron(double[] x, double[] w, double b) {
    	double z = 0.0;
    	
    	for(int i = 0; i < x.length; i++) {
    		z += x[i] * w[i];
    	}
    	
    	z += b;
    	
    	// activation function
    	return z > 0 ? 1.0 : 0.0;
	}
	
	// AND gate
	static double and(double x1, double x2) {
		return neuron(new double[] {x1, x2}, new double[] {1, 1}, -1);
	}
	
	// OR gate
	static double or(double x1, double x2) {
		return neuron(new double[] {x1, x2}, new double[] {1, 1}, 0);
	}
	
	// NOR gate
	static double nor(double x1, double x2) {
		return neuron(new double[] {x1, x2}, new double[] {-1, -1}, 1);
	}
	
	// NAND gate
	static double nand(double x1, double x2) {
		return neuron(new double[] {x1, x2}, new double[] {-1, -1}, 2);
	}
	
	// XOR gate
	static double xor(double x1, double x2) {
		return and(or(x1, x2), nand(x1, x2));
	}
	
	// XNOR gate
	static double xnor(double x1, double x2) {
		return or(and(x1, x2), nor(x1, x2));
	}
	
    public static void main( String[] args )
    {
    	for(int i = 0; i < 4; i++) {
    		double x1 = i / 2;
    		double x2 = i % 2;
    		
    		double output = xnor(x1, x2);
    		
    		System.out.printf("%d%d\t%d\n", (int)x1, (int)x2, (int)output);
    	}
    }
}
