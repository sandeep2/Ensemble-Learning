package ensemblelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.classifiers.trees.J48;

class AdaBoost {
	
	
	private final String train_file, test_file;

    public AdaBoost(String train_file, String test_file) {
        this.train_file = train_file;
        this.test_file = test_file;
    }
	
	protected final void AdaBoostDTClassify() throws IOException
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
		
		AdaBoostM1 m1 = new AdaBoostM1();
	    m1.setClassifier(new J48());
	    m1.buildClassifier(traininstance);
		
		Evaluation em1 = new Evaluation(traininstance);
		em1.evaluateModel(m1,testinstance);
		
		String summarynaive = em1.toSummaryString("\nResults: AdaBoost\n======================\n", false);
		System.out.println(summarynaive);
	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}	
	}
	
}
