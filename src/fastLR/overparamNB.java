package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class overparamNB extends LR {

	public overparamNB(Instances instances, boolean regularization, double lambda) {
		
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

		/* Convert counts to probabilities */
		for (int c = 0; c < nc; c++) {
			probabilities[c] = Math.log(Math.max(SUtils.MEsti(counts[c], N, nc), 1e-75));
		}

		for (int c = 0; c < nc; c++) {

			for (int u = 0; u < n; u++) {
				for (int uval = 0; uval < paramsPerAtt[u]; uval++) { 

					int pos = getNominalPosition(u, (int) uval, c);
					probabilities[pos] = Math.log(Math.max(SUtils.MEsti(counts[pos], counts[c], paramsPerAtt[u]), 1e-75));
				}
			}				
		}

		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
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

	public double computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {
		// Nothing here
		return 0;
	}

	@Override
	public void computeHessian(int i, double[] probs) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void computeHv(double[] s, double[] Hs) {
		// TODO Auto-generated method stub
		
	}

}
