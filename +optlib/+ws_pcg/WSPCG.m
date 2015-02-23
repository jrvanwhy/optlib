% This class represents a warm-starting preconditioned conjugate-gradient method
% for solving SPD systems of linear equations.

classdef WSPCG < handle
	methods
		% Constructor -- initializes this PCG algorithm.
		% This requires (an estimate of) the diagonal of the coefficient matrix.
		function this = WSPCG(M_Diag)
			this.precond = optlib.bfgs.LBFGSMat(M_Diag);
		end

		% This function solves the linear system to the given tolerance.
		% Note that this starts with an inital guess of 0
		% The exact algorithm used is the preconditioned conjugate gradient algorithm
		% off Wikipedia, which was chosen because it exactly matches the algorithm in
		% Eigen's ConjugateGradient implementation.
		%
		% Parameters:
		%     MxFcn Matrix-vector multiply function for the coefficient matrix
		%     b     Rhs of the linear system
		%     tol   Convergence tolerance; is finished when |r|_2 <= tol
		%
		% Returns:
		%     x     The approximate solution to the system.
		%
		function x = solve(this, MxFcn, b, tol)
			Mres  = b;                   % Residual of the original system
			Pres  = this.precond \ Mres; % Residual of the preconditioned system
			sdir  = Pres;                % Search direction
			RdotP = dot(Mres, Pres);     % Dot product of the residuals
			x     = zeros(numel(b), 1);  % The solution vector.

			% Update the tolerance to make it scale-invariant
			% Yup, I'm copying Eigen here...
			tol = tol*tol * dot(b, b);

			% Loop up to N times, where N is the size of the system to be solved.
			% In exact arithmetic, this would solve it exactly
			for iter = 1:numel(b)
				M_sdir  = MxFcn(sdir);               % Compute the M * step direction product
				alpha   = RdotP / (sdir.' * M_sdir); % Step length
				x       = x + alpha * sdir;          % Perform the update
				Mres    = Mres - alpha * M_sdir;     % Update the residual

				% Check our termination condition
				if dot(Mres, Mres) <= tol
					break
				end

				% Continue with updating values for the next iteration
				oldRdotP = RdotP;
				Pres     = this.precond \ Mres;
				RdotP    = dot(Mres, Pres);
				beta     = RdotP / oldRdotP;
				sdir     = Pres + beta * sdir;
			end

			iter
		end
	end

	properties
		precond % The preconditioner for this system (LBFGS)
	end
end
