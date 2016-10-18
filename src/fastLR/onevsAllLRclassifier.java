package fastLR;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import Utils.SUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.Bagging;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

public class onevsAllLRclassifier extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;
	private String m_S = "overparamLR"; 					// -S (overparamLR, vanillaLR)

	private boolean m_Discretization	 	= false; 			// -D
	private boolean m_MVerb 					= false; 			// -V		
	private boolean m_Regularization      = false;            // -R
	private double m_Lambda = 0.001;                           // -L

	private boolean m_DoStacking		 	= false; 			// -G

	private boolean m_ClassSpecification = false;          // -C
	private int classToSkip = 0;

	private String m_O = "QN";                                       // -O (QN, CG, GD, Tron, SGD)

	private double[] probs;	

	private LR[] lr;
	private LR lrMeta;

	private int N;
	private int nc;

	private weka.filters.supervised.attribute.Discretize m_Disc = null;

	private FastVector fvWekaAttributes = null;

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		Instances  m_DiscreteInstances = null;

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(instances);
			System.out.println("Applying Discretization Filter - dodo LR");
			m_DiscreteInstances = weka.filters.Filter.useFilter(instances, m_Disc);
			System.out.println("Done");

			m_Instances = new Instances(m_DiscreteInstances);
			m_DiscreteInstances = new Instances(m_DiscreteInstances, 0);
		} else {
			m_Instances = new Instances(instances);
			instances = new Instances(instances, 0);
		}

		N = m_Instances.numInstances();
		nc = m_Instances.numClasses();

		// Remove instances with missing class
		m_Instances.deleteWithMissingClass();

		if (m_S.equalsIgnoreCase("overparamLR")) {

			lr = new LR[nc];
			Instances[] tempInstancesX = new Instances[nc];

			Instances genericHeader = getHeader(m_Instances);

			for (int c = 0; c < nc; c++) {
				/* Create a local copy of modified instances */
				tempInstancesX[c] = new Instances(genericHeader, 0);
				updateHeader(tempInstancesX[c], c, m_Instances);

				/* Create C different classifiers */				
				lr[c] = new overparamLR(tempInstancesX[c], m_Regularization, m_Lambda, m_O);
				lr[c].train();
			}

		} else if (m_S.equalsIgnoreCase("vanillaLR")) {

			lr = new LR[nc];
			Instances[] tempInstancesX = new Instances[nc];

			Instances genericHeader = getHeader(m_Instances);

			for (int c = 0; c < nc; c++) {
				/* Create a local copy of modified instances */
				tempInstancesX[c] = new Instances(genericHeader, 0);
				updateHeader(tempInstancesX[c], c, m_Instances);

				int pos = 0;
				int N = tempInstancesX[c].numInstances();
				for (int i = 0; i < N; i++) {
					int x_C = (int) tempInstancesX[c].instance(i).classValue();

					if (x_C == 0) {
						pos++;
					}
				}
				int neg = N - pos;

				/* Create C different classifiers */				
				lr[c] = new vanillaLR(tempInstancesX[c], m_Regularization, m_Lambda, m_O);

				double eps = 0.001;
				double primal_solver_tol = eps * Math.max(Math.min(pos, neg), 1) / N; 
				lr[c].setEps(primal_solver_tol);

				lr[c].train();
			}

		} else {
			System.out.println("m_S value should be from set: {overparamLR, vanillaLR, overparamNB, vanillaNB, overparamWC, vanillaWC}");
		}

		if (m_DoStacking) {

			double[][] outputProbs = new double[N][nc+1];

			for (int i = 0; i < m_Instances.numInstances(); i++) {
				Instance inst = m_Instances.instance(i);
				int x_C = (int) inst.classValue();

				double[] probs = new double[2];
				for (int c = 0; c < nc; c++) {
					probs = lr[c].predict(inst);
					SUtils.exp(probs);

					outputProbs[i][c] = probs[0];
				}
				outputProbs[i][nc] = x_C;
			}

			/* Convert outputProbs to Arff format */
			Attribute[] attList = new Attribute[nc + 1];
			for (int c = 0; c < nc; c++) {
				attList[c] = new Attribute("att"+c);
			}
			// Declare the class attribute along with its values
			FastVector fvClassVal = new FastVector(2);
			for (int c = 0; c < nc; c++) {
				fvClassVal.addElement(c+"");
			}
			attList[nc] = new Attribute("theClass", fvClassVal);

			// Declare the feature vector
			fvWekaAttributes = new FastVector(nc + 1);
			for (int c = 0; c < nc; c++) {
				fvWekaAttributes.addElement(attList[c]);
			}
			fvWekaAttributes.addElement(attList[nc]);

			// Create an empty training set
			Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);
			// Set class index
			isTrainingSet.setClassIndex(nc);


			for (int i = 0; i < N; i++)  {
				Instance iExample = new DenseInstance(nc + 1);
				for (int c = 0; c < nc; c++) {
					iExample.setValue((Attribute)fvWekaAttributes.elementAt(c), outputProbs[i][c]);
				}
				iExample.setValue((Attribute)fvWekaAttributes.elementAt(nc), outputProbs[i][nc]);
				// add the instance
				isTrainingSet.add(iExample);
			}

			System.out.println("All Done with Creating file for Meta LR");

			lrMeta = new overparamLR(isTrainingSet, false, m_Lambda, m_O);
			lrMeta.setMaxIterations(50);
			lrMeta.train();

			System.out.println("Done training Meta LR");
		}

		// free up some space
		m_Instances = new Instances(m_Instances, 0);
	}


	@Override
	public double[] distributionForInstance(Instance instance) {

		if (m_Discretization) {
			synchronized(m_Disc) {	
				m_Disc.input(instance);
				instance = m_Disc.output();
			}
		}

		double[] classProbs = new double[2];
		double[] prediction = new double[nc];

		for (int c = 0; c < nc; c++) {
			classProbs = lr[c].predict(instance);
			SUtils.exp(classProbs);

			prediction[c] = classProbs[0];
		}

		probs = new double[nc];

		if (m_DoStacking) {
			/* Convert Prediction into Inst */
			Instance iExample = new DenseInstance(nc + 1);
			for (int c = 0; c < nc; c++) {
				iExample.setValue((Attribute)fvWekaAttributes.elementAt(c), prediction[c]);
			}
			
			probs = lrMeta.predict(iExample);
			SUtils.exp(probs);

		} else {
			int winner = SUtils.maxLocationInAnArray(prediction);
			probs[winner] = 1;
		}

		return probs;
	}	

	public static Instances getHeader(Instances instances) {
		Instances header = null;
		int n = instances.numAttributes() - 1;
		ArrayList<Attribute> attlist = new ArrayList<Attribute>(n + 1);

		for (int i = 0; i < n; i++) {
			attlist.add(instances.attribute(i));
		}

		String className = instances.classAttribute().name();
		List<String> classNamesList = new ArrayList<String>(2);
		for (int i = 0; i < 2; i++) {
			classNamesList.add(i+"");
		}
		Attribute classAtt = new Attribute(className, classNamesList);

		attlist.add(classAtt);

		header = new Instances(instances.relationName(), attlist, 0);
		header.setClassIndex(n);

		return header;
	}

	public static void updateHeader(Instances header, int classVal, Instances instances) {

		int n = instances.numAttributes();

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);
			int x_C = (int) inst.classValue();

			double[] instanceValues = new double[n];
			for (int ii = 0; ii < n - 1; ii++) {
				instanceValues[ii] = inst.value(ii);
			}

			if (x_C == classVal) {
				instanceValues[n - 1] = 0;	
			} else {
				instanceValues[n - 1] = 1;
			}

			DenseInstance denseInstance = new DenseInstance(1.0, instanceValues);
			header.add(denseInstance);
		}

	}


	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {

		m_Discretization = Utils.getFlag('D', options);
		m_MVerb = Utils.getFlag('V', options);

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		String SO = Utils.getOption('O', options);
		if (SO.length() != 0) {
			m_O = SO;
		}

		m_Regularization = Utils.getFlag('R', options);
		if (m_Regularization) {
			String SL = Utils.getOption('L', options);
			if (SL.length() != 0) {
				m_Lambda = Double.parseDouble(SL);
			}	
		}

		m_ClassSpecification = Utils.getFlag('K', options);
		if (m_ClassSpecification) {
			String SL = Utils.getOption('C', options);
			if (SL.length() != 0) {
				classToSkip = Integer.parseInt(SL);
			}	
		}

		m_DoStacking  = Utils.getFlag('G', options);

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new onevsAllLRclassifier(), argv);
	}

}
