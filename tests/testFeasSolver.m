% Clean things up
clear all

%% Test 1: Basic bound-constrained feasibility problem.
prob  = Nlp;
x     = prob.addVariable(0, 1, 2);
optim = FeasSolver(prob);
optim.export
optim.solve
assert(x.solution >= 1)
assert(x.solution <= 2)
