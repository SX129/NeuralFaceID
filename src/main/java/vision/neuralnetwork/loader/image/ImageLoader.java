package vision.neuralnetwork.loader.image;

import java.io.DataInputStream;
import java.io.FileInputStream;

import vision.neuralnetwork.loader.BatchData;
import vision.neuralnetwork.loader.Loader;
import vision.neuralnetwork.loader.MetaData;

public class ImageLoader implements Loader{
	private String imageFileName;
	private String labelFileName;
	private int batchSize;
	
	private DataInputStream dsImages;
	private DataInputStream dsLabels;
	
	public ImageLoader(String imageFileName, String labelFileName, int batchSize) {
		this.imageFileName = imageFileName;
		this.labelFileName = labelFileName;
		this.batchSize = batchSize;
	}

	@Override
	public MetaData open() {
		try {
			dsImages = new DataInputStream(new FileInputStream(imageFileName));
		} catch (Exception e) {
			throw new LoaderException("Error opening image file: " + imageFileName, e);
		}
		
		try {
			dsLabels = new DataInputStream(new FileInputStream(labelFileName));
		} catch (Exception e) {
			throw new LoaderException("Error opening label image file: " + labelFileName, e);
		}
		
		return null;
	}

	@Override
	public void close() {
		try {
			dsImages.close();
		} catch (Exception e) {
			throw new LoaderException("Error closing image file: " + imageFileName, e);
		}
		
		try {
			dsLabels.close();
		} catch (Exception e) {
			throw new LoaderException("Error closing label image file: " + labelFileName, e);
		}
	}

	@Override
	public MetaData getMetaData() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public BatchData readBatch() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
}
