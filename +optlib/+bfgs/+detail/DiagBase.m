classdef DiagBase < handle
	methods
		function this = DiagBase(diag_vals)
			this.diag_vals = diag_vals(:);
		end

		function out = mldivide(this, vec)
			out = vec ./ this.diag_vals;
		end

		function out = mtimes(this, vec)
			out = this.diag_vals .* vec;
		end
	end

	properties
		diag_vals % The values on the diagonal of this approximation
	end
end
