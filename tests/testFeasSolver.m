%% Test 1: Basic bound-constrained feasibility problem.
prob  = Nlp;
x     = prob.addVariable(0, 1, 2);
optim = FeasSolver(prob);
optim.export
optim.solve
assert(x.solution >= 1)
assert(x.solution <= 2)

%% Test 2: Add in an equality constraint -- which is satisfied anyway by interior-point methods due to bounds choice.
prob  = Nlp;
x     = prob.addVariable(0, 1, 2);
prob.addConstraint(1.5, x, 1.5)
optim = FeasSolver(prob);
optim.export
optim.solve
assert(x.solution == 1.5)

%% Test 3: Problem with an equality constraint not midway between the upper and lower bounds
prob = Nlp;
x    = prob.addVariable(0, -inf, inf);
prob.addConstraint(1, x, 1);
optim = FeasSolver(prob);
optim.export
optim.solve
assert(x.solution == 1)
