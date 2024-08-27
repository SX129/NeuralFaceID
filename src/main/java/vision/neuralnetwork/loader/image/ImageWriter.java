package vision.neuralnetwork.loader.image;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;

import vision.neuralnetwork.loader.BatchData;

public class ImageWriter {

	public static void main(String[] args) {	
		if(args.length == 0) {
			System.out.println("Usage: [app] <MNIST DATA DIRECTORY>");
			return;
		}
		
		File dir = new File(args[0]);
		
		if (!dir.isDirectory()) {
			try {
				System.out.println( "Invalid directory: " + dir.getCanonicalPath());
			} catch (Exception e) {
				e.printStackTrace();
			}
			return;
		}
		
		String directory = args[0];
		
		new ImageWriter().run(directory);
	}
	
	public void run(String directory) {		
		final String trainImages = String.format("%s%s%s", directory, File.separator, "train-images.idx3-ubyte");
		final String trainLabels = String.format("%s%s%s", directory, File.separator, "train-labels.idx1-ubyte");
		final String testImages = String.format("%s%s%s", directory, File.separator, "t10k-images.idx3-ubyte");
		final String testLabels = String.format("%s%s%s", directory, File.separator, "t10k-labels.idx1-ubyte");
		
		int batchSize = 900;
		
		ImageLoader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
		ImageLoader testLoader = new ImageLoader(testImages, testLabels, batchSize);
		
		ImageLoader loader = testLoader;
		ImageMetaData metaData = loader.open();
				
		for(int i = 0; i < metaData.getNumberBatches(); i++) {
            BatchData batchData = testLoader.readBatch();
            String montagePath = String.format("montage%d.jpg", i);
            
            System.out.println("Writing: " + montagePath);
            
            var montage = new BufferedImage(900, 900, BufferedImage.TYPE_BYTE_GRAY);
            
            try {
                ImageIO.write(montage, "jpg", new File(montagePath));
            }catch(Exception e) {
                e.printStackTrace();
            }
		}
		
		loader.close();
	}

}
