package fastLR;

import Utils.SUtils;

import optimize.DifferentiableFunction;
import optimize.FunctionValues;

import optimize.LBFGSBException;
import optimize.Result;
import optimize.StopConditions;

import optimize.Minimizer;
import optimize.MinimizerGD;
import optimize.MinimizerTron;
import optimize.MinimizerCG;

import weka.core.Instance;
import weka.core.Instances;

public abstract class LR {

	protected Instances instances;

	protected double[] parameters;
	protected double[] counts;
	protected double[] probabilities;
	protected double[] gradients;

	protected double[][][] D;
	protected double[] Dbin;

	protected int N;
	protected int nc;
	protected int n;

	protected int np = 0;

	protected int[] paramsPerAtt;
	protected int[] startPerAtt;

	protected boolean[] isNumericTrue;

	protected boolean regularization = false;
	protected double lambda = 0.001;

	protected String m_O = "QN";
	protected boolean is_Verbose = true;

	ObjectiveFunction function_to_optimize;

	int maxIterations = 10000;	

	private String name;

	public LR(Instances instances, boolean regularization, double lambda, String m_O, String name) {

		this.name = name;
		this.instances = instances;

		this.N = instances.numInstances();
		this.n = instances.numAttributes() - 1;
		this.nc = instances.numClasses();

		this.m_O = m_O;

		this.regularization = regularization;
		this.lambda = lambda;

		isNumericTrue = new boolean[n];
		paramsPerAtt = new int[n];

		function_to_optimize = new ObjectiveFunction();
	}

	class ObjectiveFunction implements DifferentiableFunction {

		@Override
		public FunctionValues getValues(double[] params) {

			double mLogNC = -Math.log(nc);
			double negLogLikelihood = 0.0;

			for (int i = 0; i < np; i++) {
				parameters[i] = params[i];
			}
			gradients = new double[np];

			for (int i = 0; i < instances.numInstances(); i++) {
				Instance inst = instances.instance(i);
				int x_C = (int) inst.classValue();
				double[] probs = predict(inst);
				negLogLikelihood += (mLogNC - probs[x_C]);
				SUtils.exp(probs);

				negLogLikelihood += computeGrad(inst, probs, x_C, gradients);
			}

			//System.out.println(negLogLikelihood);
			return new FunctionValues(negLogLikelihood, gradients);
		}

		@Override
		public double fun(double[] point) {

			double mLogNC = -Math.log(nc);
			double f = 0.0;
			
			for (int i = 0; i < np; i++) {
				parameters[i] = point[i];
			}

			for (int i = 0; i < instances.numInstances(); i++) {
				Instance inst = instances.instance(i);
				int x_C = (int) inst.classValue();
				double[] probs = predict(inst);
				f += (mLogNC - probs[x_C]);
			}

			return f;
		}

		@Override
		public void grad(double[] point, double[] grad) {

			double mLogNC = -Math.log(nc);
			double negLogLikelihood = 0.0;
			
			for (int i = 0; i < np; i++) {
				parameters[i] = point[i];
			}

			for (int i = 0; i < instances.numInstances(); i++) {
				Instance inst = instances.instance(i);
				int x_C = (int) inst.classValue();
				double[] probs = predict(inst);
				negLogLikelihood += (mLogNC - probs[x_C]);
				SUtils.exp(probs);

				negLogLikelihood += computeGrad(inst, probs, x_C, grad);
				
				computeHessian(i, probs);
			}

		}

		@Override
		public void Hv(double[] s, double[] Hs) {
			computeHv(s, Hs);
		}

		@Override
		public int get_nr_variable() {
			return np;
		}

	};

	public abstract double[] predict(Instance inst);
	public abstract double computeGrad(Instance inst, double[] probs, int x_C, double[] gradients);
	public abstract void computeHessian(int i, double[] probs);
	public abstract void computeHv(double[] s, double[] Hs);


	public void train() {

		if (m_O.equalsIgnoreCase("QN")) {

			double maxGradientNorm = 1e-32;

			Minimizer alg = new Minimizer();
			StopConditions sc = alg.getStopConditions();
			sc.setMaxGradientNorm(maxGradientNorm);
			sc.setMaxIterations(maxIterations);

			Result result;
			try {		
				if (is_Verbose) {
					System.out.println();
					System.out.print("fx_QN_" + name + " = [");
					alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});
					result = alg.run(function_to_optimize, parameters);
					System.out.println("];");
					//System.out.println(result);
					System.out.println("NoIter = " + result.iterationsInfo.iterations); System.out.println();
				} else {
					result = alg.run(function_to_optimize, parameters);
					System.out.println("NoIter = " + result.iterationsInfo.iterations);
					//System.out.println(result);
				}
			} catch (LBFGSBException e) {
				e.printStackTrace();
			}

		} else if (m_O.equalsIgnoreCase("CG")) {

			MinimizerCG alg = new optimize.MinimizerCG();
			alg.setMaxIterations(maxIterations);

			Result result;

			if (is_Verbose) {
				System.out.print("fx_CG_" + name + " = [");
				result = alg.run(function_to_optimize, parameters);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		} else if (m_O.equalsIgnoreCase("GD")) {

			MinimizerGD alg = new MinimizerGD();
			alg.setMaxIterations(maxIterations);
			Result result;	

			if (is_Verbose) {
				System.out.print("fx_GD_" + name + " = [");
				result = alg.run(function_to_optimize, parameters);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		} else if (m_O.equalsIgnoreCase("Tron")) {

			MinimizerTron alg = new MinimizerTron();
			alg.setMaxIterations(maxIterations);
			Result result;	

			if (is_Verbose) {
				System.out.print("fx_Tron_" + name + " = [");
				result = alg.run(function_to_optimize, parameters);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		} else {
			System.out.println("Only QN, CG, GD and Tron are implemented");
			System.exit(-1);
		}

	}

	public void setModel(double[] newArray) {
		for (int i = 0; i < np; i++) {
			parameters[i] = newArray[i];
		}
	}

	public double[] getModel() {
		return parameters;
	}

	public int getNP() {
		return np;
	}

	public int getNumericPosition(int u, int c) {
		return startPerAtt[u] + (paramsPerAtt[u] * c);
	}
	public int getNominalPosition(int u, int uval, int c) {
		return startPerAtt[u] + ((paramsPerAtt[u] * c) + uval);
	}
	public int getNominalPosition(int u, int uval, int c, int[] localStartPerAtt) {
		return localStartPerAtt[u] + ((paramsPerAtt[u] * c) + uval);
	}


}
