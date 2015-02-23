% This uses symbolics to solve the given unconstrained minimization problem

function soln = sym_minunc(obj, sym_x, x0)
	% Symbolic derivations
	jobj = jacobian(obj,  sym_x);
	hess = jacobian(jobj, sym_x);
	sym_vec = sym('sym_minunc_v', [numel(sym_x) 1]); % Vector for the Hessian * vector multiply function
	hMult   = hess * sym_vec;
	hDiag   = diag(hess);

	% Generate the anonymous functions
	obj_fcn   = matlabFunction(obj,   'vars', {sym_x});
	jobj_fcn  = matlabFunction(jobj,  'vars', {sym_x});
	hMult_fcn = matlabFunction(hMult, 'vars', {sym_x, sym_vec});
	hDiag_fcn = matlabFunction(hDiag, 'vars', {sym_x});

	% Create and call the solver
	solver = optlib.minunc.WSPcgNwtn;
	soln = solver.solve(obj_fcn, jobj_fcn, hMult_fcn, hDiag_fcn, x0);
end
