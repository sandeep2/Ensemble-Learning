package ensemblelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;

/**
*
* @author Pavitra
*/
class DecisionTree {
	
	private final String train_file, test_file;

    public DecisionTree(String train_file, String test_file) {
        this.train_file = train_file;
        this.test_file = test_file;
    }
	
    protected final void DTClassify() throws IOException
	{
		BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
		BufferedReader testdata = new BufferedReader(new FileReader(this.test_file));
		Instances traininstance = new Instances(traindata);
		Instances testinstance = new Instances(testdata);
		traindata.close();
		testdata.close();
		//setting class attribute
		traininstance.setClassIndex(traininstance.numAttributes() - 1);
		testinstance.setClassIndex(testinstance.numAttributes() - 1);
		try
		{
			 // train classifier
			Classifier cls = new J48();
			cls.buildClassifier(traininstance);
			// evaluate classifier and print some statistics
			Evaluation eval = new Evaluation(traininstance);
			eval.evaluateModel(cls, testinstance);
			System.out.println(eval.toSummaryString("\nResults: Decision Tree\n=====================\n", false));
		}
		catch(Exception e)
		{
			System.out.println("An exception occured: "+ e);
		}
	}
}
