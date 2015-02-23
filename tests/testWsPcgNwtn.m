clear all

syms x y

obj = (1 - x)^2 + 100 * (y - x^2)^2;
sym_x = [x; y];
x0    = [-1; 10];

soln = sym_minunc(obj, sym_x, x0)
