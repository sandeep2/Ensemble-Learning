package ensemblelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;

class BaggingMethod {

	private final String train_file, test_file;

    public BaggingMethod(String train_file, String test_file) {
        this.train_file = train_file;
        this.test_file = test_file;
    }
	
	protected final void BaggingDTClassify() throws IOException
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
		Bagging b = new Bagging();
//		AdaBoostM1 m1 = new AdaBoostM1();
	    b.setClassifier(new J48());
	    b.buildClassifier(traininstance);
		
		Evaluation eb = new Evaluation(traininstance);
		eb.evaluateModel(b,testinstance);
		
		String summarynaive = eb.toSummaryString("\nResults: Bagging\n======================\n", false);
		System.out.println(summarynaive);
	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}	
	}
	
}
