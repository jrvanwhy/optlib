clear all
close all
clc

dims = 30000000;

solver = optlib.minunc.WSPcgNwtn;
x0     = 2*ones(dims, 1);

solver.ws_pcg.lbfgs_mem = 2;
solver.ws_pcg.en_dbg    = true;

soln = solver.solve(@rosenObj, @rosenJObj, @rosenHMult, @rosenHDiag, x0);
