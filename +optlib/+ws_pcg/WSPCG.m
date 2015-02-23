% This class represents a warm-starting preconditioned conjugate-gradient method
% for solving SPD systems of linear equations.
% It contains a few hooks (such as minimum encountered curvature output) for optimization
% usage

classdef WSPCG < handle
	methods
		% Adds in a preconditioner update pair.
		% This will check if the update is valid (dot(x, Mx) > 0)
		% and ignore the pair if not
		function addPUpdate(this, x, Mx)
			dotP = dot(x, Mx);

			% Abort if not an acceptable update
			if dotP <= 0
				return
			end

			this.pcache_x(   :, end+1) = x;
			this.pcache_Mx(  :, end+1) = Mx;
			this.pcache_dotP(:, end+1) = dotP;
		end

		% This function solves the linear system to the given tolerance.
		% Note that this starts with an inital guess of 0
		% The exact algorithm used is the preconditioned conjugate gradient algorithm
		% off Wikipedia, which was chosen because it exactly matches the algorithm in
		% Eigen's ConjugateGradient implementation.
		% Note that this will deal any nonpositive entries of MDiag for the user,
		% and if the first step fails it will do a L-BFGS step instead.
		%
		% Parameters:
		%     MxFcn  Matrix-vector multiply function for the coefficient matrix
		%     b      Rhs of the linear system
		%     MDiag  Diagonal of the coefficient matrix. May be empty if restarting after a failed first iteration
		%     tol    Convergence tolerance; is finished when |r|_2 <= tol
		%
		% Returns:
		%     x      The approximate solution to the system.
		%
		function x = solve(this, MxFcn, b, MDiag, tol)
			this.upPrecond(MDiag);

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
				M_sdir  = MxFcn(sdir);     % Compute the M * step direction product
				MxT_x   = sdir.' * M_sdir; % Dot product of step and M * step, copied for the bfgs update

				% Check for nonpositive curvature and abort if it was detected
				if MxT_x <= 0
					% If this was the first iteration, just use the preconditioner
					% to solve for x
					if iter <= 1
						x = Pres;
					end

					break
				end

				alpha   = RdotP / MxT_x;         % Step length
				x       = x + alpha * sdir;      % Perform the update
				Mres    = Mres - alpha * M_sdir; % Update the residual

				% Cache things for the next L-BFGS update, if we're not already at the memory limit
				lbfgs_mem = 10;
				if size(this.pcache_x, 2) < lbfgs_mem-1 % Note that we save 1 update for the delta-gradients update before the next solve
					this.pcache_x(   :, end+1) = sdir;
					this.pcache_Mx(  :, end+1) = M_sdir;
					this.pcache_dotP(:, end+1) = MxT_x;
				end

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
		end

		% This handles the adjustment & updating of the preconditioner
		function upPrecond(this, MDiag)
			% Use the L-BFGS diagonal to fix the nonpositive entries of MDiag (if present),
			% unless this is the first solution, in which case we just set the nonpositive entries to 1
			if isempty(this.precond)
				% No existing preconditioner, set nonpositive entries to 1
				MDiag(MDiag <= 0) = 1;
			else
				% A preconditioner exists.
				pDiag             = this.precond.getDiag;
				MDiag(MDiag <= 0) = pDiag(pDiag <= 0);
			end

			% Initialize the preconditioner using our adjusted diagonal.
			this.precond = optlib.bfgs.LBFGSMat(MDiag);

			% Create updates using the cached values.
			for iter = 1:size(this.pcache_x, 2)
				this.precond.addUpdate(this.pcache_x(:, iter), this.pcache_Mx(:, iter), this.pcache_dotP(:, iter))
			end

			% Empty the preconditioner cache for next time.
			this.pcache_x    = [];
			this.pcache_Mx   = [];
			this.pcache_dotP = [];
		end
	end

	properties
		% The preconditioner itself
		precond

		% Preconditioner-related cached values
		pcache_x
		pcache_Mx
		pcache_dotP
	end
end
