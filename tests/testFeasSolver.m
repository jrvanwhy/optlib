%% Test 1: Basic bound-constrained feasibility problem.
clear all
prob  = Nlp;
x     = prob.addVariable(0, 1, 2);
optim = FeasSolver(prob);
optim.export
optim.solve
assert(x.solution >= 1)
assert(x.solution <= 2)

%% Test 2: Add in an equality constraint -- which is satisfied anyway by interior-point methods due to bounds choice.
clear all
prob  = Nlp;
x     = prob.addVariable(0, 1, 2);
prob.addConstraint(1.5, x, 1.5)
optim = FeasSolver(prob);
optim.export
optim.solve
assert(abs(x.solution - 1.5) < sqrt(eps))

%% Test 3: Problem with an equality constraint not midway between the upper and lower bounds
clear all
prob = Nlp;
x    = prob.addVariable(0, -inf, inf);
prob.addConstraint(1, x, 1)
optim = FeasSolver(prob);
optim.export
optim.solve
assert(abs(x.solution - 1) < sqrt(eps))

%% Test 4: First nonlinear problem. Actually a bit difficult -- Newton's method fails with this initial guess, even damped
clear all
prob = Nlp;
x    = prob.addVariable(0, -inf, inf);
y    = prob.addVariable(0, -inf, inf);
prob.addConstraint(2, x + y, 2)
prob.addConstraint(1, x * y, 1)
optim = FeasSolver(prob);
optim.export
optim.solve
assert(abs(x.solution - 1) < sqrt(eps))
assert(abs(y.solution - 1) < sqrt(eps))

%% Test 5: Sequentially solve a series of equations. Similar to the last, but it shouldn't accidentally stumble across the solution in 1 iteration
clear all
prob = Nlp;
x    = prob.addVariable(0, -inf, inf, 'Length', 10);
expr = 1;
for iter = 1:x.length
	expr = expr * ind(x, iter);
	prob.addConstraint(1, expr, 1)
end
optim = FeasSolver(prob);
optim.export
optim.solve
assert(all(abs(x.solution - 1) < sqrt(eps)))

%% Test 6: First general inequality-constrained problem
clear all
prob = Nlp;
x    = prob.addVariable(0, -inf, inf);
y    = prob.addVariable(0, -inf, inf);
prob.addConstraint(0,    x,     inf)
prob.addConstraint(-inf, x + y,  -1)
optim = FeasSolver(prob);
optim.export
optim.solve
assert(x.solution >= 0)
assert(x.solution + y.solution <= -1)

%% Test 7:
clear all
prob = Nlp;
x = prob.addVariable(0, -inf, inf, 'Length', 2);
prob.addConstraint(1, x.initial + x.final, inf);
optim = FeasSolver(prob);
optim.export
optim.solve
assert(sum(x.solution) >= 1)
