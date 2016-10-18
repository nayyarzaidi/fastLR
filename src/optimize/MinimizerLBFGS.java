package optimize;

import java.io.PrintStream;

public class MinimizerLBFGS {

	// Max iterations
	private int maxIterations = 1000;

	private FunctionValues fv = null;
	private int totalFunctionEvaluations = 0;	 

	private boolean verbose = false;

	public Result run(DifferentiableFunction fun_obj, double[] w, double eps) {
		
		if (verbose)
			System.out.println("All Done");

		IterationsInfo info = null;
		info = new IterationsInfo(0, totalFunctionEvaluations, IterationsInfo.StopType.MAX_ITERATIONS, null);	

		double f = 0.0;
		double[] g = null;
		Result result = new Result(w, f, g, info);
		return result;
	}

	public void setMaxIterations(int m_MaxIterations) {
		maxIterations = 	m_MaxIterations;
	}

}


