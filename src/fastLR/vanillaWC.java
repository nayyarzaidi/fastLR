package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class vanillaWC extends LR {

	public vanillaWC(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "wv");

		for (int u = 0; u < n; u++) {
			if (instances.attribute(u).isNominal()) {
				isNumericTrue[u] = false;
				paramsPerAtt[u] = instances.attribute(u).numValues();
			} else if (instances.attribute(u).isNumeric()) {
				System.out.println("Can't handle numeric attributes");
				System.exit(-1);
			}
		}

		startPerAtt = new int[n];

		np = nc;
		for (int u = 0; u < n; u++) {
			startPerAtt[u] += np;
			np += (paramsPerAtt[u] * (nc));
		}

		parameters = new double[np];
		counts = new double[np];
		
		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);
		Arrays.fill(counts, 0.0);
	}

	public void train() {

		for (int i = 0; i < N; i++) {
			Instance inst = instances.instance(i);
			int classVal = (int) inst.classValue();

			counts[classVal]++;

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					int pos = getNominalPosition(u, (int) uval, classVal);
					counts[pos]++;
				}
			}

		}

		probabilities = new double[np];
		double m = 1.0;

		/* Convert counts to probabilities */
		for (int c = 0; c < nc; c++) {
			//probabilities[c] =  Math.log((counts[c] + m/nc)/ (counts[nc - 1] + m/nc));
			probabilities[c] = Math.log(Math.max(SUtils.MEsti(counts[c], N, nc), 1e-75));
		}

		for (int c = 0; c < nc; c++) {
			for (int u = 0; u < n; u++) {
				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 

					int pos = getNominalPosition(u, (int) uval, c);
					int pos_LC = getNominalPosition(u, (int) uval, nc - 1);

					//probabilities[pos] = Math.log(((counts[pos] + m/paramsPerAtt[u]) / (counts[pos_LC] + m/paramsPerAtt[u])) * ((counts[nc - 1] + m) / (counts[c] + m)) );		
					probabilities[pos] = Math.log(Math.max(SUtils.MEsti(counts[pos], counts[c], paramsPerAtt[u]), 1e-75));
				}
			}				
		}
		
//		/* Re-allocate parameters and probabilities */
//		for (int i = 0; i < np; i ++) {
//			counts[i] = probabilities[i];
//		}
//		
//		int[] oldstartPerAtt = new int[n];
//		for (int u = 0; u < n; u++) {
//			oldstartPerAtt[u] = startPerAtt[u];
// 		}
//		
//		np = nc - 1;
//		startPerAtt = new int[n];
//		for (int u = 0; u < n; u++) {
//			startPerAtt[u] += np;
//			np += (paramsPerAtt[u] * (nc - 1));
//		}
//
//		System.out.println("Modifying Model to be of Size: " + np);
//		parameters = new double[np];
//		probabilities = new double[np];
//		
//		Arrays.fill(parameters, 0.0);
//		Arrays.fill(probabilities, 0.0);
//		
//		for (int c = 0; c < nc - 1; c++) {
//			probabilities[c] = counts[c];
//		}
//		for (int c = 0; c < nc - 1; c++) {
//			for (int u = 0; u < n; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 
//
//					int posOld = getNominalPosition(u, (int) uval, c, oldstartPerAtt);
//					int pos = getNominalPosition(u, (int) uval, c);
//
//					probabilities[pos] = counts[posOld];		
//				}
//			}				
//		}
//		
//		counts = null;
		
		super.train();
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc - 1; c++) {
			probs[c] = parameters[c] * probabilities[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);
					int pos = getNominalPosition(u, (int) uval, c);
					probs[c] += parameters[pos] * probabilities[pos];
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		double  negReg = 0.0;

		if (regularization) {

			for (int c = 0; c < nc - 1; c++) {
				negReg += lambda/2 * (parameters[c] * probabilities[c] * parameters[c] * probabilities[c]);
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * probabilities[c] 
						+ (lambda * parameters[c] * probabilities[c]);
			}

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					for (int c = 0; c < nc - 1; c++) {
						int pos = getNominalPosition(u, (int) uval, c);
						negReg += lambda/2 * (parameters[pos] * probabilities[c] * parameters[pos] * probabilities[c]);
						gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * probabilities[c]
								+ (lambda * parameters[pos] * probabilities[c]);
					}
				}	
			}

		} else {

			for (int c = 0; c < nc - 1; c++) {
				gradients[c] -= (SUtils.ind(c, x_C) - probs[c]) * probabilities[c];
			}

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					for (int c = 0; c < nc - 1; c++) {
						int pos = getNominalPosition(u, (int) uval, c);
						gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]) * probabilities[pos];
					}
				}
			}
		}

	}

	@Override
	public void computeHessian(int i, double[] probs) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void computeHv(double[] s, double[] Hs) {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public double regularizeFunction() {
		double f = 0.0;
		for (int i = 0; i < np; i++) {
			f += lambda/2 * parameters[i] * parameters[i];
		}
		return f;
	}

	@Override
	public void regularizeGradient(double[] grad) {
		for (int i = 0; i < np; i++) {
			grad[i] += lambda * parameters[i];
		}
	}


}
