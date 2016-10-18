package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class vanillavanillaLR extends LR {

	public vanillavanillaLR(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "vv");

		for (int u = 0; u < n; u++) {
			if (instances.attribute(u).isNominal()) {
				isNumericTrue[u] = false;
				paramsPerAtt[u] = instances.attribute(u).numValues();

				if (paramsPerAtt[u] != 1)
					paramsPerAtt[u] = paramsPerAtt[u] - 1;

			} else if (instances.attribute(u).isNumeric()) {
				isNumericTrue[u] = true;
				paramsPerAtt[u] = 1;
			}
		}

		startPerAtt = new int[n];

		np = nc  - 1;
		for (int u = 0; u < n; u++) {
			startPerAtt[u] += np;
			if (instances.attribute(u).isNominal()) {
				np += (paramsPerAtt[u] * (nc - 1));
			} else if (instances.attribute(u).isNumeric()) {
				np += (1 * (nc - 1));
			}
		}

		parameters = new double[np];
		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);

		if (m_O.equalsIgnoreCase("Tron")) {
			if (nc == 2) {
				Dbin = new double[N];
			} else {
				D = new double[N][nc - 1][nc - 1];
			}
		}
	}

	public void train() {
		super.train();
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc - 1; c++) {
			probs[c] = parameters[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						probs[c] += (parameters[pos] * uval);
					} else {
						if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, c);
							probs[c] += parameters[pos];
						}
					}
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		for (int c = 0; c < nc - 1; c++) {
			gradients[c] -= (SUtils.ind(c, x_C) - probs[c]);
		}

		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				for (int c = 0; c < nc - 1; c++) {
					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]) * uval;
					} else {
						if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, c);
							gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]);
						}
					}
				}
			}
		}

	}

	public void computeHessian(int i, double[] probs) {

		if (nc == 2) {
			Dbin[i] = (1 - probs[0]) * probs[0];
		} else {

			for (int c1 = 0; c1 < nc - 1; c1++) {
				for (int c2 = 0; c2 < nc - 1; c2++) {

					if (c1 == c2) {
						D[i][c1][c2] = (1 - probs[c1]) * probs[c1];
					} else {
						D[i][c1][c2] = -probs[c1] * probs[c2];
					}

				}			
			}

		}

	}

	@Override
	public void computeHv(double[] s, double[] Hs) {

		double[] wa = new double[N * (nc - 1)];
		double[] wa2 = new double[N * (nc - 1)];

		int[] offset = new int[nc - 1];
		int index = 0;
		for (int c = 0; c < nc - 1; c++) {
			offset[c] = index;
			index += N;
		}

		//Xv(s, wa);
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);

			for (int c = 0; c < nc - 1; c++) {

				wa[i + offset[c]] += s[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						wa[i] += (s[pos] * uval);
					} else {
						if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, c);
							wa[i + offset[c]] += s[pos];
						}
					}

				}

			}
		}

		//D[i] * wa[i];
		for (int i = 0; i < instances.numInstances(); i++) {
			if (nc == 2) {
				wa2[i] = (Dbin[i] * wa[i]);
			} else {

				for (int c1 = 0; c1 < nc - 1; c1++) {					
					for (int c2 = 0; c2 < nc - 1; c2++) {
						wa2[i + offset[c1]] += (D[i][c1][c2] * wa[i + offset[c2]]);
					}
				}

			}
		}

		//XTv(wa, Hs);
		for (int i = 0; i < np; i++) {
			Hs[i] = 0;
		}

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);

			for (int c = 0; c < nc - 1; c++) {

				Hs[c] += wa2[i + offset[c]];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						Hs[pos] += (wa2[i + offset[c]] * uval);
					} else {
						if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, c);
							Hs[pos] += wa2[i + offset[c]];
						}
					}

				}

			}
		}	

		//s[i] + Hs[i];
		for (int i = 0; i < np; i++) {
			Hs[i] = s[i] + Hs[i];
		}

	}


}
