package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class overparamWC extends LR {

	public overparamWC(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "wop");

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

		parameters = new double[np];
		counts = new double[np];

		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);
		Arrays.fill(counts, 0.0);

		if (m_O.equalsIgnoreCase("Tron")) {
			D = new double[N][nc][nc];
		}
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

		super.train();
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
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

		for (int c = 0; c < nc; c++) {
			gradients[c] -= (SUtils.ind(c, x_C) - probs[c]) *  probabilities[c];
		}

		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				for (int c = 0; c < nc; c++) {
					int pos = getNominalPosition(u, (int) uval, c);
					gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]) * probabilities[pos];
				}
			}
		}

	}

	@Override
	public void computeHessian(int i, double[] probs) {
		for (int c1 = 0; c1 < nc; c1++) {
			for (int c2 = 0; c2 < nc; c2++) {

				if (c1 == c2) {
					D[i][c1][c2] = (1 - probs[c1]) * probs[c1];
				} else {
					D[i][c1][c2] = -probs[c1] * probs[c2];
				}

			}			
		}

	}

	@Override
	public void computeHv(double[] s, double[] Hs) {
		double[] wa = new double[N * nc];
		double[] wa2 = new double[N * nc];

		int[] offset = new int[nc];
		int index = 0;
		for (int c = 0; c < nc; c++) {
			offset[c] = index;
			index += N;
		}

		//Xv(s, wa);
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);

			for (int c = 0; c < nc; c++) {

				wa[i + offset[c]] += (s[c] * probabilities[c]);

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					int pos = getNominalPosition(u, (int) uval, c);
					wa[i + offset[c]] += (s[pos] * probabilities[pos]);
				}

			}
		}

		//D[i] * wa[i];
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int c1 = 0; c1 < nc; c1++) {					
				for (int c2 = 0; c2 < nc; c2++) {
					wa2[i + offset[c1]] += (D[i][c1][c2] * wa[i + offset[c2]]);
				}
			}
		}

		//XTv(wa, Hs);
		for (int i = 0; i < np; i++) {
			Hs[i] = 0;
		}

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);

			for (int c = 0; c < nc; c++) {

				Hs[c] += (wa2[i + offset[c]] * probabilities[c]);

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					int pos = getNominalPosition(u, (int) uval, c);
					Hs[pos] += (wa2[i + offset[c]] * probabilities[pos]);
				}

			}
		}	

		//s[i] + Hs[i];
		for (int i = 0; i < np; i++) {
			Hs[i] = s[i] + Hs[i];
		}

	}
	
	@Override
	public double regularizeFunction() {
		double f = 0.0;
		for (int i = 0; i < np; i++) {
			//f += lambda/2 * (parameters[i] * probabilities[i]) * (parameters[i] * probabilities[i]);
			f += lambda/2 * (parameters[i]) * (parameters[i]);
		}
		return f;
	}

	@Override
	public void regularizeGradient(double[] grad) {
		for (int i = 0; i < np; i++) {
			//grad[i] += (lambda * parameters[i]  * probabilities[i]);
			grad[i] += (lambda * parameters[i]);
		}
	}

}
