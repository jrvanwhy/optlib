%% Test 1: Large multi-variable Rosenbrock function minimization with a poor initial guess.
x = sym('x', [100 1]);
obj   = sum(100*(x(2:end) - x(1:end-1).^2).^2 + (1 - x(1:end-1)).^2);
soln  = sym_minunc(obj, x, 10 * ones(numel(x), 1));
assert(all(abs(soln - ones(numel(soln), 1)) < 1e-6))
