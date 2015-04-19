package ensemblelearning;

import java.io.IOException;
import weka.core.Instances;

/**
 *
 * @author RavitejaSomisetty
 */
public interface Classifier {
     Instances getData() throws IOException;
     Instances runClassifier() throws ArrayIndexOutOfBoundsException;
}