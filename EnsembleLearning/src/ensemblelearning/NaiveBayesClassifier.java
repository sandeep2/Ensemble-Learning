package ensemblelearning;

import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
*
* @author Pavitra
*/
class NaiveBayesClassifier {
	
	private final String train_file, test_file;

    public NaiveBayesClassifier(String train_file, String test_file) {
        this.train_file = train_file;
        this.test_file = test_file;
    }
	
	protected final void NBClassify() throws IOException
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
		Classifier cnaive = (Classifier)new NaiveBayes();
		cnaive.buildClassifier(traininstance);
		
		Evaluation enaive = new Evaluation(traininstance);
		enaive.evaluateModel(cnaive,testinstance);
		
		String summarynaive = enaive.toSummaryString("\nResults: Naive Bayes\n======================\n", false);
		System.out.println(summarynaive);
	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}	
	}
}
