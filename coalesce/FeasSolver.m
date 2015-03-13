% COALESCE interface for solving feasibility problems.

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
			clear feasCeq feasJCeq feasC feasJC

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

			% Generate the inequality constraints functions. This is similar to the equality constraint export above
			% First, check for a lack of inequality constraints (again, we need to special case if there are no inequality constraints)
			if any(~conIsEq)
				ineqCons = this.nlp.constraint(~conIsEq);
				c        = [];
				for iter = 1:nnz(~conIsEq)
					if isfinite(ineqCons(iter).lowerBound)
						c = [ c; ineqCons(iter).lowerBound - ineqCons(iter).expression ];
					end

					if isfinite(ineqCons(iter).upperBound)
						c = [ c; ineqCons(iter).expression - ineqCons(iter).upperBound ];
					end
				end
			else
				c = ConstantNode.empty;
			end
			% Generate the constraint function itself
			gen = MatlabFunctionGenerator({'var'}, {'c'}, 'feasC');
			gen.writeHeader
			gen.writeExpression(c, 'c')
			gen.writeFooter
			% Generate the c Jacobian function
			gen = MatlabFunctionGenerator({'var'}, {'jc'}, 'feasJC');
			gen.writeHeader
			[iJ, jJ, sJ] = c.jacobian;
			gen.writeIndex(iJ, 'iJ')
			gen.writeIndex(jJ, 'jJ')
			gen.writeExpression(sJ, 'sJ')
			fprintf(gen.fid, '\tjc = sparse(iJ, jJ, sJ, %d, %d);\n', nVars, sum([c.length]));
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
			cVal     = feasC(cur_x);
			meritVal = this.meritFcn(ceqVal, cVal);

			% Check for failed initial constraint evaluation
			if ~isfinite(meritVal)
				error('Initial constraint evaluation failed!')
			end

			% Optimization main loop
			for iter = 1:this.maxIter
				% Evaluate the derivative for use in assembling the LP subproblem
				jCeq = feasJCeq(cur_x);
				jC   = feasJC(cur_x);

				% Put together the LP subproblem

				% Objective first. This is just a sum of slack variables
				objjac = [ zeros(numel(cur_x), 1)
				           ones(numel(ceqVal)+numel(cVal), 1) ];

				% Put together the slack variable definition constraints
				A = this.robustVCat([ jCeq, -eye(numel(ceqVal)) ], ...
				                    [-jCeq, -eye(numel(ceqVal)) ], ...
				                    [ jC,   -eye(numel(cVal))   ]);

				b = [ -ceqVal
				       ceqVal
				      -cVal ];

				% Check for corner cases
				if isempty(b)
					A = [];
				end

				% Compute the lower bounds. Since the LP is relative, these are shifted with respect
				% to the real upper and lower bounds.
				lp_lb = [ this.nlp.variableLowerBound(:) - cur_x
				          -inf(numel(ceqVal), 1)
				           zeros(numel(cVal), 1) ];

				lp_ub = [ this.nlp.variableUpperBound(:) - cur_x
				          abs(ceqVal)
				          max(0, cVal) ];

				% Compute the step and desired objective function decrease
				[lp_step, desobjchg] = this.callLPSolver(objjac, A, b, lp_lb, lp_ub);

				% Pull out just the x step
				step = lp_step(1:numel(cur_x));

				% Line search!
				slen = 1;
				while true
					new_x       = cur_x + slen * step;
					newCeqVal   = feasCeq(new_x);
					newCVal     = feasC(new_x);
					newMeritVal = this.meritFcn(newCeqVal, newCVal);

					if newMeritVal <= meritVal + slen * desobjchg / 2
						break
					end

					slen = slen / 2;
				end

				% Update the current point and other associated values
				cur_x    = new_x;
				ceqVal   = newCeqVal;
				cVal     = newCVal;
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
		function val = meritFcn(this, ceqVal, cVal)
			val = sum(abs(ceqVal)) + sum(max(0, cVal));
		end

		% Utility function to more robustly perform vertical concatenation
		% (i.e. it ignores empty matrices)
		function out = robustVCat(this, varargin)
			% Compute the width of the output matrix, for initialization
			width = 0;
			for iter = 1:numel(varargin)
				width = max(width, size(varargin{iter}, 2));
			end

			% Build up the output matrix through input-wise concatenation
			out = zeros(0, width);
			for iter = 1:numel(varargin)
				% Skip empty inputs
				if isempty(varargin{iter})
					continue
				end

				out = [ out; varargin{iter} ];
			end
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
