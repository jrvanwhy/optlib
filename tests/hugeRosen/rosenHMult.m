function out = rosenHMult(x, v)
	out = [ v(1) * (1200*x(1)^2 - 400*x(2) + 2) - 400 * v(2) * x(1)
	        v(2:end-1) .* (1200 * x(2:end-1).^2 - 400 * x(3:end) + 202) - 400 * v(1:end-2) .* x(1:end-2) - 400 * v(3:end) .* x(2:end-1)
	        200 * v(end) - 400 * v(end-1) * x(end-1) ];
end
