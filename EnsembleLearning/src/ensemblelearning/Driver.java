/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ensemblelearning;


/**
 *
 * @author RavitejaSomisetty
 */
public class Driver {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        PreProcessor p = new PreProcessor("E:\\Spring2015\\Data Mining\\Ensemble Learning Project\\Ensemble-Learning\\census-income.data",
                "E:\\Spring2015\\Data Mining\\Ensemble Learning Project\\Ensemble-Learning\\census-income-preprocessed.arff");
        p.run();
    }

}
