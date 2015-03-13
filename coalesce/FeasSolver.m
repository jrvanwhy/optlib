% COALESCE interface for solving feasibility problems.
%
% TODO:
%     * Inequality constraints (other than bounds)
%     * Robustness (i.e. nonfinite constraint values)

classdef FeasSolver < Solver
	methods
		function this = FeasSolver(prob)
			% Call superclass constructor
			this = this@Solver(prob);
		end

		function export(this)
			% User feedback
			disp('Exporting FeasSolver functions')

			% Clear out old functions so that MATLAB will recognize our new ones
			clear feasCeq feasJCeq

			% These are used repeatedly later
			nVars = this.nlp.numberOfVariables;

			% Identify equality constraints
			conIsEq = [];
			for iter = 1:numel(this.nlp.constraint)
				conIsEq(iter) = all(this.nlp.constraint(iter).lowerBound == this.nlp.constraint(iter).upperBound);
			end
			conIsEq = logical(conIsEq);

			% Generate the equality constraints functions
			% First, check for a lack of equality constraints -- we need to special case or the referencing will fail
			if any(conIsEq)
				eqCons = this.nlp.constraint(conIsEq);
				for iter = 1:nnz(conIsEq)
					ceq(iter) = eqCons(iter).expression - eqCons(iter).lowerBound;
				end
			else
				ceq = ConstantNode.empty;
			end
			% Generate the constraint function itself
			gen = MatlabFunctionGenerator({'var'}, {'ceq'}, 'feasCeq');
			gen.writeHeader
			gen.writeExpression(ceq, 'ceq')
			gen.writeFooter
			% Generate the ceq Jacobian function
			gen = MatlabFunctionGenerator({'var'}, {'jceq'}, 'feasJCeq');
			gen.writeHeader
			[iJeq, jJeq, sJeq] = ceq.jacobian;
			gen.writeIndex(iJeq, 'iJeq')
			gen.writeIndex(jJeq, 'jJeq')
			gen.writeExpression(sJeq, 'sJeq')
			fprintf(gen.fid, '\tjceq = sparse(iJeq, jJeq, sJeq, %d, %d);\n', nVars, sum([ceq.length]));
			gen.writeFooter
		end

		function solve(this)
			% User feedback
			disp('Beginning FeasSolver solve')

			% Refresh MATLAB's caches, so as to avoid calling outdated functions
			rehash

			% Formulate the initial guess. Clamp as necessary between the lower and upper bounds
			% (so that we always satisfy the bounds throughout the solution process).
			cur_x = min(max(this.nlp.initialGuess(:), this.nlp.variableLowerBound(:)), this.nlp.variableUpperBound(:));

			% Was the solve successful? This gets set to true if the termination criteria are met.
			success = false;

			% Initialize saved values
			ceqVal   = feasCeq(cur_x);
			meritVal = this.meritFcn(ceqVal);

			% Optimization main loop
			for iter = 1:this.maxIter
				% Evaluate the derivative for use in assembling the LP subproblem
				jCeq = feasJCeq(cur_x);

				% Put together the LP subproblem

				% Objective first. This is just a sum of slack variables
				objjac = [ zeros(numel(cur_x), 1)
				           ones(numel(ceqVal), 1) ];

				% Put together the slack variable definition constraints
				A = [ jCeq, -eye(numel(ceqVal))
				     -jCeq, -eye(numel(ceqVal)) ];

				b = [ -ceqVal
				       ceqVal ];

				% Check for corner cases
				if isempty(b)
					A = [];
				end

				% Compute the lower bounds. Since the LP is relative, these are shifted with respect
				% to the real upper and lower bounds. TODO: Add in the bounds on the slack variables...
				lp_lb = [ this.nlp.variableLowerBound(:) - cur_x
				          -inf(numel(ceqVal), 1) ];

				lp_ub = [ this.nlp.variableUpperBound(:) - cur_x
				          abs(ceqVal) ];

				% Compute the step and desired objective function decrease
				[lp_step, desobjchg] = this.callLPSolver(objjac, A, b, lp_lb, lp_ub);

				% Pull out just the x step
				step = lp_step(1:numel(cur_x));

				% Line search!
				slen = 1;
				while true
					new_x       = cur_x + slen * step;
					newCeqVal   = feasCeq(new_x);
					newMeritVal = this.meritFcn(newCeqVal);

					if newMeritVal <= meritVal + slen * desobjchg / 2
						break
					end

					slen = slen / 2;
				end

				% Update the current point and other associated values
				cur_x    = new_x;
				ceqVal   = newCeqVal;
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
		function [soln,objval] = callLPSolver(this, objjac, A, b, lb, ub)
			[soln,objval] = linprog(objjac, A, b, [], [], lb, ub, [], this.lpoptions);
		end

		% Merit function for the optimization.
		function val = meritFcn(this, ceqVal)
			val = sum(abs(ceqVal));
		end
	end

	properties
		% Solver parameters
		maxIter   = 100       % Maximum number of iterations during the solution
		tolerance = sqrt(eps) % Termination tolerance for the solver

		% Options for the LP solver
		lpoptions = optimoptions(@linprog, 'Display', 'off');
	end
end
