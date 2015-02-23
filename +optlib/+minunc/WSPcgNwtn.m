% This class is a warm-startable preconditioned conjugate gradient-based
% Newton method for minimizing a function.
% It uses a line search to globalize Newton's method.

classdef WSPcgNwtn < handle
	methods
		% Constructor. Just initializes things.
		function this = WSPcgNwtn
			this.ws_pcg = optlib.ws_pcg.WSPCG;
		end

		% Curvature-adjusted hessian*vector multiply function.
		%
		% This multiplies by H + a*I, where H is the actual Hessian
		function out = hMult_adj(this, vec)
			out = this.hMult(this.cur_x, vec) + this.cvture_adj;
		end

		% Backtracking linesearch
		function [f1,act_step] = linesearch_bt(this, obj, jobj_val, f0, x_step)
			fdiff = jobj_val * x_step;
			slen  = 1;

			for iter = 1:10
				act_step = slen * x_step;
				new_x = this.cur_x + act_step;
				f1    = obj(new_x);

				if f1 < f0 + fdiff * slen/10
					break
				end

				slen = slen / 2;
			end

			this.cur_x = new_x;
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
			this.cur_x = x0;

			% Send hMult_adj() hMult
			this.hMult = hMult;

			% Initialize the function value and its jacobian
			fval     = obj(this.cur_x);
			jobj_val = jobj(this.cur_x);

			% Main loop. We have a maximum iteration limit.
			for iter = 1:1000
				% Compute the Newton system right hand side
				Nrhs = -jobj_val(:);

				% Compute the diagonal of the adjusted Hessian
				hDiag_val = hDiag(this.cur_x) + this.cvture_adj;

				% Try to solve it. If the solution fails,
				% try again with a different curvature adjustment.
				pcg_tol = sqrt(eps);
				[x_step, mindotP] = this.ws_pcg.solve(@this.hMult_adj, Nrhs, hDiag_val, pcg_tol);
				this.up_cvture(mindotP);
				if isempty(x_step)
					[x_step, mindotP] = this.ws_pcg.solve(@this.hMult_adj, Nrhs, hDiag_val, pcg_tol);
					this.up_cvture(mindotP);
				end

				% Line search!
				[fval,x_step] = this.linesearch_bt(obj, jobj_val, fval, x_step);

				% Compute the new jacobian value, then check our termination condition
				jobj_new = jobj(this.cur_x);
				if norm(jobj_new) < sqrt(eps)
					soln = this.cur_x;
					break
				end

				% Give the preconditioner a newer hint...
				this.ws_pcg.addPUpdate(x_step, jobj_new - jobj_val);

				% Copy over the new jacobian value because we don't need the old one any more
				jobj_val = jobj_new;
			end

			disp(['Iterations: ' num2str(iter)])
		end

		% Curvature update function
		function up_cvture(this, mindotP)
			% Check if the need for a curvature adjustment has disappeared
			if mindotP > this.cvture_adj
				this.cvture_adj = 0;
				return
			end

			% Update the curvature adjustment to target a certain value
			cvture_tgt = sqrt(eps);
			this.cvture_adj = this.cvture_adj + cvture_tgt - mindotP;
		end
	end

	properties
		cur_x          % The current point in the optimization. Used by callback functions.
		hMult          % Handle for the Hessian multiply function. Used to send hMult from solve() to hMult_adj()
		ws_pcg         % The warm-startable linear system solver we're using
		cvture_adj = 0 % Curvature adjustment (>= 0)
	end
end
