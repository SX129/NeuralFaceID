package vision.neuralnetwork.loader.test;

import static org.junit.Assert.*;

import vision.neuralnetwork.loader.BatchData;
import vision.neuralnetwork.loader.Loader;
import vision.neuralnetwork.loader.MetaData;

import org.junit.Test;

public class TestTestLoader {

	@Test
	public void test() {
		int batchSize = 32;
		Loader testLoader = new TestLoader(60_000, batchSize);
		MetaData metaData = testLoader.open();
		
		for (int i = 0; i < metaData.getNumberBatches(); i++) {
			BatchData batchData = testLoader.readBatch();
			
			assertTrue(batchData != null);
			
			int itemsRead = metaData.getItemsRead();
			
			assertTrue(itemsRead == batchSize);
		}
	}

}
