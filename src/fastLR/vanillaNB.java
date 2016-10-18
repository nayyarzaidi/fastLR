package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class vanillaNB extends LR {

	public vanillaNB(Instances instances, boolean regularization, double lambda) {

		super(instances, regularization, lambda,  "", "");

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
			np += (paramsPerAtt[u] * nc);
		}

		counts = new double[np];
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
			probabilities[c] =  Math.log((counts[c] + m/nc)/ (counts[nc - 1] + m/nc));
		}

		for (int c = 0; c < nc; c++) {
			for (int u = 0; u < n; u++) {
				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 

					int pos = getNominalPosition(u, (int) uval, c);
					int pos_LC = getNominalPosition(u, (int) uval, nc - 1);

					probabilities[pos] = Math.log(((counts[pos] + m/paramsPerAtt[u]) / (counts[pos_LC] + m/paramsPerAtt[u])) * 
							((counts[nc - 1] + m) / (counts[c] + m)) );		
				}
			}				
		}
		


		
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc - 1; c++) {
			probs[c] = probabilities[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					int pos = getNominalPosition(u, (int) uval, c);
					probs[c] += probabilities[pos];
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	@Override
	public void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {
		// Nothing here
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
		// Nothing here
		return 0;
	}

	@Override
	public void regularizeGradient(double[] grad) {
		// Nothing here
	}
	
}





//for (int c = 0; c < nc - 1; c++) {
//
//	for (int u = 0; u < n; u++) {
//		for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 
//
//			int pos = getNominalPosition(u, (int) uval, c);
//			int posL = getNominalPosition(u, paramsPerAtt[u] - 1, c);
//			
//			probabilities[pos] -= probabilities[posL];				
//		}
//	}				
//}

//		/* Convert counts to probabilities */
//		for (int c = 0; c < nc; c++) {
//			probabilities[c] = Math.max(SUtils.MEsti(counts[c], N, nc), 1e-75);
//		}
//
//		for (int c = 0; c < nc; c++) {
//
//			for (int u = 0; u < n; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 
//
//					int pos = getNominalPosition(u, (int) uval, c);
//					probabilities[pos] = Math.max(SUtils.MEsti(counts[pos], counts[c], paramsPerAtt[u]), 1e-75);
//				}
//			}				
//		}

//		/* Convert counts to probabilities */
//		for (int c = 0; c < nc; c++) {
//			probabilities[c] = (1 - n) * Math.log(counts[c] / counts[nc - 1]);
//
//		}
//
//		for (int c = 0; c < nc; c++) {
//
//			for (int u = 0; u < n; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 
//
//					int pos = getNominalPosition(u, (int) uval, c);
//					int posL = getNominalPosition(u, paramsPerAtt[u] - 1, c);
//
//					int pos_LC = getNominalPosition(u, (int) uval, nc - 1);
//					int posL_LC = getNominalPosition(u, paramsPerAtt[u] - 1, nc - 1);
//
//					probabilities[pos] = Math.log((counts[pos] / counts[posL]) / (counts[pos_LC] / counts[posL_LC]) );
//				}
//			}				
//		}

//		/* Normalize each attribute by its last attribute value */
//		for (int c = 0; c < nc; c++) {
//
//			for (int u = 0; u < n; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 
//
//					int pos = getNominalPosition(u, (int) uval, c);
//					int posL = getNominalPosition(u, paramsPerAtt[u] - 1, c);
//					
//					probabilities[pos] /= probabilities[posL];
//				}
//			}				
//		}
//		
//		/* Normalize by class values */
//		
//		for (int c = 0; c < nc; c++) {
//			probabilities[c] /= probabilities[nc - 1];
//		}
//		
//		for (int c = 0; c < nc; c++) {
//
//			for (int u = 0; u < n; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 
//
//					int pos = getNominalPosition(u, (int) uval, c);
//					int posL = getNominalPosition(u, (int) uval, nc - 1);
//					
//					probabilities[pos] /= probabilities[posL];
//				}
//			}				
//		}
//		
//		/* Take log */
//		for (int i = 0; i < np; i++) {
//			probabilities[i] = Math.log(probabilities[i]);
//		}
