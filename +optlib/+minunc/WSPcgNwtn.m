% This class is a warm-startable preconditioned conjugate gradient-based
% Newton method for minimizing a function.
% It uses a line search to globalize Newton's method.

classdef WSPcgNwtn < handle
	methods
		% Constructor. Just initializes things.
		function this = WSPcgNwtn
			this.ws_pcg            = optlib.ws_pcg.WSPCG;
			this.ws_pcg.dbg_indent = 4;
		end

		% Backtracking linesearch
		function [new_x,f1,act_step,slen] = linesearch_bt(this, obj, jobj_val, f0, cur_x, x_step)
			fdiff = jobj_val * x_step;
			slen  = 1;

			for iter = 1:20
				act_step = slen * x_step;
				new_x = cur_x + act_step;
				f1    = obj(new_x);

				if f1 < f0 + fdiff * slen/10
					break
				end

				slen = slen / 2;
			end
		end

		% Minimization function!
		%
		% Parameters:
		%     obj   Handle to the objective function
		%     jobj  Handle to the objective jacobian function
		%     hMult Handle to the Hession * vector multiply function
		%     hDiag Hessian diagonal function
		%     x0    Initial guess
		%
		% Returns:
		%     soln  The solution to the optimization problem.
		function soln = solve(this, obj, jobj, hMult, hDiag, x0)
			% The current optimization variable vector
			cur_x = x0;

			% Initialize the function value and its jacobian
			fval        = obj(cur_x);
			jobj_val    = jobj(cur_x);
			jobj_nrmsqr = dot(jobj_val, jobj_val);

			% The tolerance for the PCG-based subproblem solver
			pcg_tol = .1;

			% Main loop. We have a maximum iteration limit.
			for iter = 1:1000
				% Compute the Newton system right hand side
				Nrhs = -jobj_val(:);

				% Set up the parameters for the PCG solution
				pcg_maxiter = numel(cur_x);
				if iter == 1 && this.ws_pcg.lbfgs_mem > 1
					pcg_maxiter = this.ws_pcg.lbfgs_mem - 1;
				end

				% Try to solve it. If the solution fails,
				% just use a steepest descent step.
				x_step = this.ws_pcg.solve(@(v) hMult(cur_x, v), Nrhs, hDiag(cur_x), pcg_tol, pcg_maxiter);

				% Line search!
				[cur_x, fval, x_step, slen] = this.linesearch_bt(obj, jobj_val, fval, cur_x, x_step);

				% Compute the new jacobian value, then check our termination condition
				jobj_new        = jobj(cur_x);
				jobj_nrmsqr_new = dot(jobj_new, jobj_new);
				if jobj_nrmsqr_new < eps
					soln = cur_x;
					break
				end

				% Give the preconditioner a newer hint...
				this.ws_pcg.addPUpdate(x_step, jobj_new - jobj_val);

				% Use the decrease in Jacobian to compute a new PCG solver tolerance
				pcg_tol = min(max(jobj_nrmsqr_new / jobj_nrmsqr, sqrt(eps)), .1);

				% Copy over the new jacobian-related values for the next iteration
				jobj_val    = jobj_new;
				jobj_nrmsqr = jobj_nrmsqr_new;
			end
		end
	end

	properties
		ws_pcg % The warm-startable linear system solver we're using
	end
end
