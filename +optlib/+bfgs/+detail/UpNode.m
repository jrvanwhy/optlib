classdef UpNode < handle
	methods
		function this = UpNode(oldTree, x, Mx)
			this.oldTree = oldTree;
			this.x       = x;
			this.Mx      = Mx;
			this.MxT_x   = Mx.' * x;
			this.Bx      = oldTree * x;
			this.xT_Bx   = x.' * this.Bx;
		end

		function out = mldivide(this, vec)
			Binv_prod = this.oldTree \ (vec - this.Mx * ((this.x.' * vec) / this.MxT_x));
			out       = Binv_prod - this.x * ((this.Mx.' * Binv_prod) / this.MxT_x) + this.x * ((this.x.' * vec) / this.MxT_x);
		end

		function out = mtimes(this, vec)
			out = this.oldTree * vec + this.Mx * ((this.Mx.' * vec) / this.MxT_x) - this.Bx * ((this.Bx.' * vec) / this.xT_Bx);
		end
	end

	properties
		oldTree % The old LBFGS data tree

		x     % The x vector from the M*x update
		Mx    % The product between the actual matrix and the step (x).
		MxT_x % Dot product of Mx and x
		Bx    % Product of the old B matrix and x
		xT_Bx % Dot product between x and Bx (curvature information)
	end
end
