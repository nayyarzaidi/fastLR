package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class overparamLR extends LR {

	public overparamLR(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "op");

		for (int u = 0; u < n; u++) {
			if (instances.attribute(u).isNominal()) {
				isNumericTrue[u] = false;
				paramsPerAtt[u] = instances.attribute(u).numValues();
			} else if (instances.attribute(u).isNumeric()) {
				isNumericTrue[u] = true;
				paramsPerAtt[u] = 1;
			}
		}

		startPerAtt = new int[n];

		np = nc;
		for (int u = 0; u < n; u++) {
			startPerAtt[u] += np;
			if (instances.attribute(u).isNominal()) {
				np += (paramsPerAtt[u] * nc);
			} else if (instances.attribute(u).isNumeric()) {
				np += (1 * nc);
			}
		}

		parameters = new double[np];
		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);

		if (m_O.equalsIgnoreCase("Tron")) {
			D = new double[N][nc][nc];
		}

	}

	public void train() {
		super.train();
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = parameters[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						probs[c] += (parameters[pos] * uval);
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						probs[c] += parameters[pos];
					}
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public void computeHessian(int i, double[] probs) {

		for (int c1 = 0; c1 < nc; c1++) {
			for (int c2 = 0; c2 < nc ; c2++) {

				if (c1 == c2) {
					D[i][c1][c2] = (1 - probs[c1]) * probs[c1];
				} else {
					D[i][c1][c2] = -probs[c1] * probs[c2];
				}

			}			
		}

	}

	public double computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		double  negLLReg = 0.0;

		if (regularization) {

			for (int c = 0; c < nc; c++) {
				negLLReg += lambda/2 * parameters[c] * parameters[c];
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]) + lambda * parameters[c];
			}

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					for (int c = 0; c < nc; c++) {
						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u, c);
							negLLReg += lambda/2 * parameters[pos] * parameters[pos];
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * uval + lambda * parameters[pos];
						} else {
							int pos = getNominalPosition(u, (int) uval, c);
							negLLReg += lambda/2 * parameters[pos] * parameters[pos];
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) + lambda * parameters[pos];
						}
					}
				}
			}

		} else {

			for (int c = 0; c < nc; c++) {
				gradients[c] += (-1) * (SUtils.ind(c, x_C) - probs[c]);
			}

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					for (int c = 0; c < nc; c++) {
						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u, c);
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * uval;
						} else {
							int pos = getNominalPosition(u, (int) uval, c);
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]);
						}
					}
				}
			}
		}


		return negLLReg;
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

				wa[i + offset[c]] += s[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);
					int pos = getNominalPosition(u, (int) uval, c);

					wa[i + offset[c]] += s[pos];
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
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);
			int x_C = (int) inst.classValue();

			Hs[x_C] += wa2[i + offset[x_C]];

			for (int c = 0; c < nc; c++) { 		

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);
					int pos = getNominalPosition(u, (int) uval, c);

					Hs[pos] += wa2[i + offset[c]];
				}

			}
		}	

		//s[i] + Hs[i];
		for (int i = 0; i < np; i++) {
			Hs[i] = s[i] + Hs[i];
		}
		
	}

}
