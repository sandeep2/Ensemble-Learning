package ensemblelearning;
import java.io.IOException;
/**
 *
 * @author RavitejaSomisetty
 */
public class Driver {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
    	
    	String original_training_file = "F:\\MSCS - CS6220 DM\\Final Project\\census-income\\census-income.data";
    	String new_training_file = "F:\\MSCS - CS6220 DM\\Final Project\\census-income\\census-income-preprocessed.arff";
    	String original_test_file = "F:\\MSCS - CS6220 DM\\Final Project\\census-income\\census-income.test";
    	String new_test_file = "F:\\MSCS - CS6220 DM\\Final Project\\census-income\\census-income-test-preprocessed.arff"; 
    	
        PreProcessor p = new PreProcessor(original_training_file,new_training_file);
        p.run();
        
        // created a new instance for imputation of the test file
        PreProcessor p_test = new PreProcessor(original_test_file,new_test_file);
        
        p_test.run();
        
        NaiveBayesClassifier nb = new NaiveBayesClassifier(new_training_file,new_test_file);
        nb.NBClassify();
        
        DecisionTree dt = new DecisionTree(new_training_file,new_test_file);
        dt.DTClassify();
        
        AdaBoost ab = new AdaBoost(new_training_file,new_test_file);
        ab.AdaBoostDTClassify();
        
        BaggingMethod bag = new BaggingMethod(new_training_file,new_test_file);
        bag.BaggingDTClassify();
    }
}
