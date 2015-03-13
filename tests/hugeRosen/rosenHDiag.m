function out = rosenHDiag(x)
	out = [ 1200*x(1)^2 - 400*x(2) + 2
	        1200 * x(2:end-1).^2 - 400*x(3:end) + 202
	        200 ];
end
