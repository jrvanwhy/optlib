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
			clear feasCons feasJCons

			% These are used repeatedly later
			nVars = this.nlp.numberOfVariables;

			% Identify equality constraints
			conIsEq = [];
			for iter = 1:numel(this.nlp.constraint)
				conIsEq(iter) = all(this.nlp.constraint(iter).lowerBound == this.nlp.constraint(iter).upperBound);
			end
			conIsEq = logical(conIsEq);

			% Generate the constraints functions
			% First, check for a lack of each constraint type constraints -- we need to special case or the referencing will fail
			if any(~conIsEq)
				c = vertcat(this.nlp.constraint(~conIsEq).expression);
			else
				c = ConstantNode.empty;
			end
			ceqShift = []; % Constant value to shift the equality constraints by
			if any(conIsEq)
				ceq      = vertcat(this.nlp.constraint(conIsEq).expression);
				ceqShift = ConstantNode(vertcat(this.nlp.constraint(conIsEq).lowerBound));
			else
				ceq = ConstantNode.empty;
			end
			% Generate the constraint function itself
			gen = MatlabFunctionGenerator({'var'}, {'c', 'ceq'}, 'feasCons');
			gen.writeHeader
			gen.writeExpression(c,   'c'  )
			gen.writeExpression(ceq, 'ceq')
			if ~isempty(ceqShift)
				gen.writeExpression(ceqShift, 'ceqShift')
				fprintf(gen.fid, '\tceq = ceq - ceqShift;\n');
			end
			gen.writeFooter
			% Generate the Jacobian function
			gen = MatlabFunctionGenerator({'var'}, {'jc', 'jceq'}, 'feasJCons');
			gen.writeHeader
			[iJ,   jJ,   sJ  ] = c.jacobian;
			[iJeq, jJeq, sJeq] = ceq.jacobian;
			gen.writeIndex(iJ,   'iJ'  )
			gen.writeIndex(jJ,   'jJ'  )
			gen.writeIndex(iJeq, 'iJeq')
			gen.writeIndex(jJeq, 'jJeq')
			gen.writeExpression(sJ,   'sJ'  )
			gen.writeExpression(sJeq, 'sJeq')
			fprintf(gen.fid, '\tjc   = sparse(iJ,   jJ,   sJ,   %d, %d);\n', sum([c.length]),   nVars);
			fprintf(gen.fid, '\tjceq = sparse(iJeq, jJeq, sJeq, %d, %d);\n', sum([ceq.length]), nVars);
			gen.writeFooter

			% Go ahead and build the constraint mappings
			ineqCons = this.nlp.constraint(~conIsEq);
			curIdx   = 1;
			for iter = 1:numel(ineqCons)
				conSize = numel(ineqCons(iter).lowerBound);

				if isfinite(ineqCons(iter).lowerBound)
					this.cIdxs   = [ this.cIdxs;    curIdx:(curIdx + conSize - 1) ];
					this.cMults  = [ this.cMults;  -ones(conSize, 1)              ];
					this.cShifts = [ this.cShifts;  ineqCons(iter).lowerBound     ];
				end

				if isfinite(ineqCons(iter).upperBound)
					this.cIdxs   = [ this.cIdxs;    curIdx:(curIdx + conSize - 1) ];
					this.cMults  = [ this.cMults;   ones(conSize, 1)              ];
					this.cShifts = [ this.cShifts; -ineqCons(iter).upperBound     ];
				end

				curIdx = curIdx + conSize;
			end
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
			[ cVal, ceqVal ] = this.evalCons(cur_x);
			meritVal         = this.meritFcn(ceqVal, cVal);

			% Check for failed initial constraint evaluation
			if ~isfinite(meritVal)
				error('Initial constraint evaluation failed!')
			end

			% Optimization main loop
			for iter = 1:this.maxIter
				% Evaluate the derivative for use in assembling the LP subproblem
				[ jC, jCeq ] = this.evalJCons(cur_x);

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
					new_x                  = cur_x + slen * step;
					[ newCVal, newCeqVal ] = this.evalCons(new_x);
					newMeritVal            = this.meritFcn(newCeqVal, newCVal);

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

		% Function to evaluate the constraints using the generated functions.
		% This performs the mapping between the COALESCE-generated inequality
		% expressions and the NLP constraint values
		function [c, ceq] = evalCons(this, x)
			% Evaluate the equality constraints and "raw" inequality constraint expressions
			[rawC, ceq] = feasCons(x);

			% Perform the inequality constraint mapping
			c = rawC(this.cIdxs) .* this.cMults + this.cShifts;
		end

		% Similar to evalCons, this evaluates the Jacobians of the constraint functions
		function [jC, jCeq] = evalJCons(this, x)
			% Similarly, we start by grabbing the "raw" jacobian
			[rawJC, jCeq] = feasJCons(x);

			% Again, we need to perform the inequality constraint mapping
			jCPos = rawJC(this.cIdxs(this.cMults > 0), :);
			jCNeg = rawJC(this.cIdxs(this.cMults < 0), :);
			jC    = sparse([], [], [], numel(this.cIdxs), size(jCeq, 2));
			jC(this.cMults > 0, :) = jCPos;
			jC(this.cMults < 0, :) = -jCNeg;
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

		% Mappings for the inequality constraints
		cIdxs   = [] % Indexes for the inequality constraints
		cMults  = [] % Signs for the inequality constraints
		cShifts = [] % Offsets for the inequality constraints (double)
	end
end
