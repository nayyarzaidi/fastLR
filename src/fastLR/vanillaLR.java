package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class vanillaLR extends LR {

	public vanillaLR(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "v");

		for (int u = 0; u < n; u++) {
			if (instances.attribute(u).isNominal()) {
				isNumericTrue[u] = false;
				paramsPerAtt[u] = instances.attribute(u).numValues();

				//if (paramsPerAtt[u] != 1)
				//	paramsPerAtt[u] = paramsPerAtt[u] - 1;

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
						//if (uval != paramsPerAtt[u]) {
						int pos = getNominalPosition(u, (int) uval, c);
						probs[c] += parameters[pos];
						//}
					}
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
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

	public double computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		double  negReg = 0.0;

		if (regularization) {

			for (int c = 0; c < nc - 1; c++) {
				negReg += lambda/2 * parameters[c] * parameters[c];
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) + lambda * parameters[c];
			}

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					for (int c = 0; c < nc - 1; c++) {
						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u, c);
							negReg += lambda/2 * parameters[pos] * parameters[pos];
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * uval + lambda * parameters[pos];
						} else {
							//if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, c);
							negReg += lambda/2 * parameters[pos] * parameters[pos];
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) + lambda * parameters[pos];
							//}
						}
					}
				}	
			}

		} else {

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
							//if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, c);
							gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]);
							//}
						}
					}
				}
			}
		}

		return negReg;
	}

	@Override
	public void computeHv(double[] s, double[] Hs) {

		double[] wa = new double[N];

		//Xv(s, wa);
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);
			int x_C = (int) inst.classValue();

			for (int c = 0; c < nc - 1; c++) {

				wa[i] += s[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						wa[i] += (s[pos] * uval);
					} else {
						//if (uval != paramsPerAtt[u]) {
						int pos = getNominalPosition(u, (int) uval, c);
						wa[i] += s[pos];
						//}
					}

				}

			}
		}

		//D[i] * wa[i];
		for (int i = 0; i < instances.numInstances(); i++) {
			wa[i] = Dbin[i] * wa[i];
		}

		//XTv(wa, Hs);
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);
			int x_C = (int) inst.classValue();

			for (int c = 0; c < nc - 1; c++) {

				Hs[c] += wa[i];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						Hs[pos] += (wa[i] * uval);
					} else {
						//if (uval != paramsPerAtt[u]) {
						int pos = getNominalPosition(u, (int) uval, c);
						Hs[pos] += wa[i];
						//}
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
