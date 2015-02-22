classdef LBFGSMat < handle
	methods
		function this = LBFGSMat(approx_diag)
			this.matTree = optlib.bfgs.detail.DiagBase(approx_diag(:));
		end

		function out = mldivide(this, vec)
			out = this.matTree \ vec;
		end

		function out = mtimes(this, vec)
			out = this.matTree * vec;
		end

		function addUpdate(this, x, Mx)
			this.matTree = optlib.bfgs.detail.UpNode(this.matTree, x, Mx);
		end
	end

	properties
		matTree % The last matTree node.
	end
end
