classdef UpNode < handle
	methods
		function this = UpNode(oldTree, x, Mx, dotP)
			this.oldTree = oldTree;
			this.x       = x;
			this.Mx      = Mx;

			if nargin < 4 || isempty(dotP)
				this.MxT_x = Mx.' * x;
			else
				this.MxT_x = dotP;
			end
		end

		% Function to compute the Bx vector and xT_Bx scalar, used as an intermedate step for a couple of
		% other calculations. If called when Bx has already been computed, then this will
		% just return with little cost
		function calcBx_xTBx(this)
			% Catch and ignore the "already done here" case.
			if ~isempty(this.Bx)
				return
			end

			this.Bx    = this.oldTree * this.x;
			this.xT_Bx = dot(this.x, this.Bx);
		end

		function vals = getDiag(this)
			% Compute the diagonal entries if this has not already been done
			if isempty(this.diag)
				this.calcBx_xTBx
				this.diag = this.oldTree.getDiag + (this.Mx.^2)/this.MxT_x - (this.Bx.^2)./this.xT_Bx;
			end

			% Return the computed values
			vals = this.diag;
		end

		function out = mldivide(this, vec)
			Binv_prod = this.oldTree \ (vec - this.Mx * ((this.x.' * vec) / this.MxT_x));
			out       = Binv_prod - this.x * ((this.Mx.' * Binv_prod) / this.MxT_x) + this.x * ((this.x.' * vec) / this.MxT_x);
		end

		function out = mtimes(this, vec)
			% These aren't always calculated, so initialize them if necessary
			this.calcBx_xTBx

			% Here's the actual calculation
			out = this.oldTree * vec + this.Mx * ((this.Mx.' * vec) / this.MxT_x) - this.Bx * ((this.Bx.' * vec) / this.xT_Bx);
		end
	end

	properties
		oldTree % The old LBFGS data tree

		x          % The x vector from the M*x update
		Mx         % The product between the actual matrix and the step (x).
		MxT_x      % Dot product of Mx and x
		Bx    = [] % Product of the old B matrix and x
		xT_Bx = [] % Dot product between x and Bx (curvature information)
		diag  = [] % Diagonal entries
	end
end
