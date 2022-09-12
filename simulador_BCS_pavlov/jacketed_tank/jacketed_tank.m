function [ f ] = jacketed_tank(t,x,modelo,uk_1)

L = x(1);
T = x(2);

Fi = uk_1(1);
Fv = uk_1(2);

At = modelo.At;
k = modelo.k;
rho = modelo.rho;
Cp = modelo.Cp;
lambda = modelo.lambda;
Ti = modelo.Ti;

f1 = ( Fi - k*sqrt(L) ) / At;
f2 = ( rho*Fi*Cp*(Ti - T) +  Fv*lambda )/ ( rho*At*L*Cp ) ;
f = [f1; f2];

end

