clear all

M = hilb(1000);
b1 = M * ones(size(M, 2), 1);
b2 = M * (1:size(M, 2)).';

wspcg = optlib.ws_pcg.WSPCG;

[x1, mindotP1] = wspcg.solve(@(x) M * x, b1, diag(M), sqrt(eps));
[x2, mindotP2] = wspcg.solve(@(x) M * x, b2, diag(M), sqrt(eps));
[x3, mindotP3] = wspcg.solve(@(x) M * x, b1, diag(M), sqrt(eps));
