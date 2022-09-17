function [Ak,Bk,Ck,Dk] = linearizacao_bcs()
x0 = sym('x0',[5 1],'real');
uk = sym('uk',[3 1],'real');
% Estados
x = [x0(1); x0(2); x0(3); x0(4); x0(5)];
% Entradas
u = [uk(1) uk(2) uk(3)];
% Constantes
g   = 9.81;   % Gravitational acceleration constant [m/s²]
Cc = 2e-5 ;   % Choke valve constant
A1 = 0.008107;% Cross-section area of pipe below ESP [m²]
A2 = 0.008107;% Cross-section area of pipe above ESP [m²]
D1 = 0.1016;  % Pipe diameter below ESP [m]
D2 = 0.1016;  % Pipe diameter above ESP [m]
hw = 1000;    % Total vertical distance in well [m]
L1 =  500;    % Length from reservoir to ESP [m]
L2 = 1200;    % Length from ESP to choke [m]
V1 = 4.054;   % Pipe volume below ESP [m3]
V2 = 9.729;   % Pipe volume above ESP [m3]
h1 = 200;
f0 = 60;      % ESP characteristics reference freq [Hz]
b1 = 1.5e9;   % Bulk modulus below ESP [Pa]
b2 = 1.5e9;   % Bulk modulus above ESP [Pa]
M  = 1.992e8; % Fluid inertia parameters [kg/m4]
rho = 950;    % Density of produced fluid [kg/mÃ?Â³]
pr = 1.26e7;  % Reservoir pressure
PI = 2.32e-9; % Well productivy index [m3/s/Pa]
mu  = 0.025;  % Viscosity [Pa*s]
Inp = 65;     % ESP motor nominal current [A]
Pnp = 1.625e5;% ESP motor nominal Power [W]
dfq_max = 0.5;   % máxima variação em f/s
dzc_max = 0.5;   % máxima variação em zc %/s
dudt_max = [dfq_max;dzc_max];
tp =[1/dfq_max;1/dzc_max];  % Actuator Response time 
CH = -0.03*mu + 1;
Cq = 2.7944*mu^4 - 6.8104*mu^3 + 6.0032*mu^2 - 2.6266*mu + 1;
Cp = -4.4376*mu^4 + 11.091*mu^3 -9.9306*mu^2 + 3.9042*mu + 1;
% SEA
% Calculo do HEAD e delta de pressão
q0 = x0(3)/Cq*(f0/x0(4)); H0 = -1.2454e6*q0^2 + 7.4959e3*q0 + 9.5970e2;
H = CH*H0*(x0(4)/f0)^2; % Head
Pp = rho*g*H;       % Delta de pressão
F1 = 0.158*((rho*L1*x0(3)^2)/(D1*A1^2))*(mu/(rho*D1*x0(3)))^(1/4);
F2 = 0.158*((rho*L2*x0(3)^2)/(D2*A2^2))*(mu/(rho*D2*x0(3)))^(1/4);
% Vazao do rezervatorio vazao da chocke
qr  = PI*(pr - x0(1));
qc  = Cc*(x0(5)/100)*sign((x0(2) - uk(3)))*sqrt(abs(x0(2) - uk(3)));
% 
P0 = -2.3599e9*q0^3 -1.8082e7*q0^2 +4.3346e6*q0 + 9.4355e4;
pin = x0(1) - rho*g*h1 - F1;  % Pintake
P = Cp*P0*(x0(4)/f0)^3;        % Potencia
I = Inp*P/Pnp;      % Corrente
% EDOs
dpbhdt = b1/V1*(qr - x0(3));
dpwhdt = b2/V2*(x0(3) - qc);
dqdt = 1/M*(x0(1) - x0(2) - rho*g*hw - F1 - F2 + Pp);
% Impondo a máxima variação das entradas
dfqdt = (uk(1) - x0(4))/tp(1);
dzcdt = (uk(2) - x0(5))/tp(2);

dxdt = [dpbhdt;dpwhdt;dqdt;dfqdt;dzcdt];    % EDO
%-------------------------------------------------
y = [pin;H;P;I;qc;qr]; % Variaveis de saída

Ak = jacobian(dxdt,x);
Bk = jacobian(dxdt,u);
Ck = jacobian(y,x); 
Dk = jacobian(y,u);

end