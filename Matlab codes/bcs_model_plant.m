function [ dxdt ] = bcs_model_plant(t,x,u)
% Estados
pbh = x(1); pwh = x(2); q = x(3); fq = x(4); zc = x(5);
% Entradas
fqref = u(1); zcref = u(2); pm = u(3);
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
f0 = 60;      % ESP characteristics reference freq [Hz]
b1 = 1.5e9;   % Bulk modulus below ESP [Pa]
b2 = 1.5e9;   % Bulk modulus above ESP [Pa]
M  = 1.992e8; % Fluid inertia parameters [kg/m4]
rho = 950;    % Density of produced fluid [kg/mÃ?Â³]
pr = 1.26e7;  % Reservoir pressure
PI = 2.32e-9; % Well productivy index [m3/s/Pa]
mu  = 0.025;  % Viscosity [Pa*s]
dfq_max = 0.5;   % máxima variação em f/s
dzc_max = 0.5;   % máxima variação em zc %/s
dudt_max = [dfq_max;dzc_max];
tp =[1/dfq_max;1/dzc_max];  % Actuator Response time 
CH = -0.03*mu + 1;
Cq = 2.7944*mu^4 - 6.8104*mu^3 + 6.0032*mu^2 - 2.6266*mu + 1;

% SEA
% Calculo do HEAD e delta de pressão
q0 = q/Cq*(f0/fq); H0 = -1.2454e6*q0^2 + 7.4959e3*q0 + 9.5970e2;
H = CH*H0*(fq/f0)^2; % Head
Pp = rho*g*H;       % Delta de pressão
F1 = 0.158*((rho*L1*q^2)/(D1*A1^2))*(mu/(rho*D1*q))^(1/4);
F2 = 0.158*((rho*L2*q^2)/(D2*A2^2))*(mu/(rho*D2*q))^(1/4);
% Vazao do rezervatorio vazao da chocke
qr  = PI*(pr - pbh);
qc  = Cc*(zc/100)*sign((pwh - pm))*sqrt(abs(pwh - pm));

% EDOs
dpbhdt = b1/V1*(qr - q);
dpwhdt = b2/V2*(q - qc);
dqdt = 1/M*(pbh - pwh - rho*g*hw - F1 - F2 + Pp);
% Impondo a máxima variação das entradas
dfqdt = (fqref - fq)/tp(1);
dzcdt = (zcref - zc)/tp(2);
dudt = zeros(1,2);
if (abs(dfqdt)>dudt_max(1))
    dudt(1) = sign(dfqdt)*dudt_max(1);
else
    dudt(1) = dfqdt;
end
   
if (abs(dzcdt)>dudt_max(2))
    dudt(2) = sign(dzcdt)*dudt_max(2);
else
    dudt(2) = dzcdt;
end

dxdt = [dpbhdt;dpwhdt;dqdt;dudt'];


end