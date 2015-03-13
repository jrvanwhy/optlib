% COALESCE interface for solving feasibility problems.
%
% TODO:
%     * Pass bounds to linear solver and the solution back.
%     * Termination test.
%     * Equality constraints
%     * Inequality constraints (other than bounds)

classdef FeasSolver < Solver
	methods
		function this = FeasSolver(prob)
			% Call superclass constructor
			this = this@Solver(prob);
		end

		function export(this)
			% User feedback
			disp('Exporting FeasSolver functions')

			% TODO...
		end

		function solve(this)
			% User feedback
			disp('Beginning FeasSolver solve')

			% Refresh MATLAB's caches, so as to avoid calling outdated functions
			rehash

			% Formulate the initial guess. Clamp as necessary between the lower and upper bounds
			% (so that we always satisfy the bounds throughout the solution process).
			cur_x = min(max(this.nlp.initialGuess, this.nlp.variableLowerBound), this.nlp.variableUpperBound);

			% Stuff the solution back into the COALESCE problem instance
			this.nlp.solution = cur_x;
		end
	end
end
