% This class is a warm-startable preconditioned conjugate gradient-based
% Newton method for minimizing a function.
% It uses a line search to globalize Newton's method.

classdef WSPcgNwtn < handle
	methods
		% Constructor. Just initializes things.
		function this = WSPcgNwtn
			this.ws_pcg = optlib.ws_pcg.WSPCG;
		end

		% Backtracking linesearch
		function [new_x,f1,act_step,slen] = linesearch_bt(this, obj, jobj_val, f0, cur_x, x_step)
			fdiff = jobj_val * x_step;
			slen  = 1;

			for iter = 1:10
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
			fval     = obj(cur_x);
			jobj_val = jobj(cur_x);

			% Steepest descent steplength multiplier
			sd_slen = 1;

			% Main loop. We have a maximum iteration limit.
			for iter = 1:1000
				% Compute the Newton system right hand side
				Nrhs = -jobj_val(:);

				% Compute the diagonal elements of the Hessian
				hDiagVals = hDiag(cur_x);
				hDiagVals(hDiagVals <= 0) = sqrt(eps);

				% Try to solve it. If the solution fails,
				% just use a steepest descent step.
				pcg_tol = sqrt(eps);
				x_step = this.ws_pcg.solve(@(v) hMult(cur_x, v), Nrhs, hDiagVals, pcg_tol);
				is_sdstep = isempty(x_step);
				if is_sdstep
					x_step = sd_slen * Nrhs;
				end

				% Line search!
				[cur_x, fval, x_step, slen] = this.linesearch_bt(obj, jobj_val, fval, cur_x, x_step);

				% Update the steepest descent step length if this was a steepest descent step
				if is_sdstep
					if slen >= 1
						sd_slen = 2 * sd_slen;
					else
						sd_slen = slen * sd_slen;
					end
				end

				% Compute the new jacobian value, then check our termination condition
				jobj_new = jobj(cur_x);
				if norm(jobj_new) < sqrt(eps)
					soln = cur_x;
					break
				end

				% Give the preconditioner a newer hint...
				this.ws_pcg.addPUpdate(x_step, jobj_new - jobj_val);

				% Copy over the new jacobian value because we don't need the old one any more
				jobj_val = jobj_new;
			end

			disp(['Iterations: ' num2str(iter)])
		end
	end

	properties
		ws_pcg % The warm-startable linear system solver we're using
	end
end
