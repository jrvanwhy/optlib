% This class is a warm-startable preconditioned conjugate gradient-based
% Newton method for minimizing a function.
% It uses a line search to globalize Newton's method.

classdef WSPcgNwtn < handle
	methods
		% Constructor. Just initializes things.
		function this = WSPcgNwtn
			this.ws_pcg = optlib.ws_pcg.WSPCG;
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
		function soln = solve(obj, jobj, hMult, hDiag, x0)
			% The current optimization variable vector
			cur_x = x0;

			% Main loop. We have a maximum iteration limit.
			for iter = 1:1000
				% Compute the Newton residual and preconditioner initialization values.
			end
		end
	end

	properties
		ws_pcg         % The warm-startable linear system solver we're using
		cvture_adj = 0 % Curvature adjustment (>= 0)
	end
end
