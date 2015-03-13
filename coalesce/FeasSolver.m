% COALESCE interface for solving feasibility problems.
%
% TODO:
%     * Equality constraints
%     * Inequality constraints (other than bounds)
%     * Robustness (i.e. nonfinite constraint values

classdef FeasSolver < Solver
	methods
		function this = FeasSolver(prob)
			% Call superclass constructor
			this = this@Solver(prob);
		end

		function export(this)
			% User feedback
			disp('Exporting FeasSolver functions')

			% Identify equality constraints
			conIsEq = [];
			for iter = 1:numel(this.nlp.constraint)
				conIsEq(iter) = all(this.nlp.constraint(iter).lowerBound == this.nlp.constraint(iter).upperBound);
			end
			conIsEq = logical(conIsEq);

			% Generate the equality constraints function
			gen = MatlabFunctionGenerator({'var'}, {'ceq'}, 'feasCeq');
			gen.writeHeader;
			% Check for a lack of equality constraints -- we need to special case or the referencing will fail
			if any(conIsEq)
				ceq = vertcat(this.nlp.constraint(conIsEq).expression - this.nlp.constraint(conIsEq).lowerBound);
			else
				ceq = ConstantNode.empty;
			end
			gen.writeExpression(ceq, 'ceq')
			gen.writeFooter
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

			% Initialize saved values
			meritVal = this.meritFcn(cur_x);

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
				slen = 1;
				while true
					new_x       = cur_x + slen * step;
					newMeritVal = this.meritFcn(new_x);

					if newMeritVal <= meritVal + slen * desobjchg / 2
						break
					end

					slen = slen / 2;
				end

				% Update the current point and other associated values
				cur_x    = new_x;
				meritVal = newMeritVal;

				% Termination check!
				if meritVal <= this.tolerance * size(this.nlp.initialGuess)
					success = true;
					break
				end
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
		% This is intended to interface to different LP solvers
		function [soln,objval] = callLPSolver(this, objjac, A, b, Aeq, beq, lb, ub)
			[soln,objval] = linprog(objjac, A, b, Aeq, beq, lb, ub, [], this.lpoptions);
		end

		% Merit function for the optimization.
		function val = meritFcn(this, x)
			val = sum(abs(feasCeq(x)));
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
