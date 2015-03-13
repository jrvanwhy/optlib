% COALESCE interface for solving feasibility problems.
%
% TODO:
%     * Termination test.
%     * Equality constraints
%     * Line search
%     * Inequality constraints (other than bounds)
%     * Robustness

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

			% Was the solve successful? This gets set to true if the termination criteria are met.
			success = false;

			% Optimization main loop
			for iter = 1:this.maxIter
				% TODO: Setup!
				objjac = zeros(numel(cur_x), 1);
				A      = [];
				b      = [];
				Aeq    = [];
				beq    = [];

				% Compute the lower bounds. Since the LP is relative, these are shifted with respect
				% to the real upper and lower bounds. TODO: Add in the bounds on the slack variables...
				lp_lb = this.nlp.variableLowerBound - cur_x;
				lp_ub = this.nlp.variableUpperBound - cur_x;

				% Compute the step and desired objective function decrease
				[step, desobjchg] = this.callLPSolver(objjac, A, b, Aeq, beq, lp_lb, lp_ub);

				% Line search!
				% TODO: This
				slen = 1;

				% Update the current point
				cur_x = cur_x + step;

				% Termination check!
				% TODO: Implement a line search, which will give us the necessary residual norm
				if iter == this.maxIter
					success = true;
					break
				end
				%if resNorm <= this.tolerance * size(this.nlp.initialGuess)
				%	success = true;
				%	break
				%end
			end

			% Stuff the solution back into the COALESCE problem instance
			this.nlp.solution = cur_x;

			% Check if we hit the iteration limit and throw an error if we did.
			if ~success && iter == this.maxIter
				error('Maximum iteration count reached!')
			end
		end

		% This calls the linear programming solver behind the scenes.
		% Assumes the initial guess is the zero vector (useful for relative step computation)
		% TODO: Try different solvers. Gotta' support them all!
		function [soln,objval] = callLPSolver(this, objjac, A, b, Aeq, beq, lb, ub)
			[soln,objval] = linprog(objjac, A, b, Aeq, beq, lb, ub, [], this.lpoptions);
		end
	end

	properties
		% Solver parameters
		maxIter   = 3         % Maximum number of iterations during the solution
		tolerance = sqrt(eps) % Termination tolerance for the solver

		% Options for the LP solver
		lpoptions = optimoptions(@linprog, 'Display', 'off');
	end
end
