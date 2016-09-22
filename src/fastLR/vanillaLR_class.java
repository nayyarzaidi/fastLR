package fastLR;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class vanillaLR_class extends LR {

	private int classToSkip;
	private int[] classIndices;

	public vanillaLR_class(Instances instances, boolean regularization, double lambda, int classToSkip, String m_O) {

		super(instances, regularization, lambda,  m_O, "vc_" + classToSkip);

		this.classToSkip = classToSkip;

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
		Arrays.fill(parameters, 0.0);

		classIndices = new int[nc];
		int j = 0;
		for (int c = 0; c < nc; c++) {
			if (c == classToSkip) {
				classIndices[c] = -1;
			} else {
				classIndices[c] = j;
				j = j + 1;
			}
		}

	}

	public void train() {
		super.train();
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			int newc = classIndices[c];

			if (newc != -1) {
				probs[c] = parameters[newc];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, newc);
						probs[c] += parameters[pos];
					} else {
						if (uval != paramsPerAtt[u]) {
							int pos = getNominalPosition(u, (int) uval, newc);
							probs[c] += parameters[pos];
						}
					}

				}	
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public double computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		double  negReg = 0.0;

		if (regularization) {

			for (int c = 0; c < nc; c++) {
				int newc = classIndices[c];
				if (newc != -1) {
					negReg += lambda/2 * parameters[newc] * parameters[newc];
					gradients[newc] += (-1) * (SUtils.ind(c, x_C) - probs[c]) + lambda * parameters[newc];
				}
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(u);

				for (int c = 0; c < nc; c++) {
					int newc = classIndices[c];
					if (newc != -1) {
						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u, newc);
							negReg += lambda/2 * parameters[pos] * parameters[pos];
							gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) * uval + lambda * parameters[pos];
						} else {
							if (uval != paramsPerAtt[u]) {
								int pos = getNominalPosition(u, (int) uval, newc);
								negReg += lambda/2 * parameters[pos] * parameters[pos];
								gradients[pos] += (-1) * (SUtils.ind(c, x_C) - probs[c]) + lambda * parameters[pos];
							}
						}
					}
				}
			}

		} else {

			for (int c = 0; c < nc; c++) {
				int newc = classIndices[c];
				if (newc != -1) {
					gradients[newc] -= (SUtils.ind(c, x_C) - probs[c]);
				}
			}

			for (int u = 0; u < n; u++) {
				double uval = inst.value(u);

				for (int c = 0; c < nc; c++) {
					int newc = classIndices[c];
					if (newc != -1) {
						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u, newc);
							gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]) * uval;
						} else {
							if (uval != paramsPerAtt[u]) {
								int pos = getNominalPosition(u, (int) uval, newc);
								gradients[pos] -= (SUtils.ind(c, x_C) - probs[c]);
							}
						}
					}
				}
			}

		}


		return negReg;
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
