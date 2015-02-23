clear all

M = hilb(1000);
b = M * ones(size(M, 2), 1);

wspcg = optlib.ws_pcg.WSPCG;

[x1, mindotP1] = wspcg.solve(@(x) M * x, b, diag(M), sqrt(eps));
[dx1,mindotP2] = wspcg.solve(@(x) M * x, b - M * x1, diag(M), .01);
