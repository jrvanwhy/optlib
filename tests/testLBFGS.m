%% Test 1: Diagonal in R^3
obj = optlib.bfgs.LBFGSMat([1 2 3]);
assert(all(obj * [1; 2; 3] == [1; 4; 9]))
assert(all(obj \ [1; 2; 3] == [1; 1; 1]))
assert(all(obj \ (obj * [16; 32; 23]) == [16; 32; 23]))
assert(all(obj.getDiag == [1; 2; 3]))
M(:, 1) = obj * [1; 0; 0];
M(:, 2) = obj * [0; 1; 0];
M(:, 3) = obj * [0; 0; 1];
assert(all(eig(M) == [1; 2; 3]))
assert(all(all(M == M.')))

%% Test 2: Single update in R^3
obj = optlib.bfgs.LBFGSMat([1 1 1]);
obj.addUpdate([1; 1; 1], [1; 2; 3])
assert(all(obj * [1; 1; 1] == [1; 2; 3]))
assert(all(obj \ [1; 2; 3] == [1; 1; 1]))
assert(all(abs(obj \ (obj * [3; 32; 23]) - [3; 32; 23]) < 1e-13))
M(:, 1) = obj * [1; 0; 0];
M(:, 2) = obj * [0; 1; 0];
M(:, 3) = obj * [0; 0; 1];
assert(all(eig(M) > 0))
assert(all(all(M == M.')))
