function [ypk] = eq_medicao(x)

    % Estados
    pbh = x(1); pwh = x(2); q = x(3); fq = x(4); zc = x(5);
    
    % Parametros
    f0 = 60;
    rho = 950;
    g   = 9.81;
    h1 = 200;
    L1 =  500;
    D1 = 0.1016;
    A1 = 0.008107;
    mu  = 0.025;
    CH = -0.03*mu + 1;
    Cq = 2.7944*mu^4 - 6.8104*mu^3 + 6.0032*mu^2 - 2.6266*mu + 1;
    Cp = -4.4376*mu^4 + 11.091*mu^3 -9.9306*mu^2 + 3.9042*mu + 1;
        
    q0 = q/Cq*(f0/fq); 
    H0 = -1.2454e6*q0^2 + 7.4959e3*q0 + 9.5970e2;
    F1 = 0.158*((rho*L1*q^2)/(D1*A1^2))*(mu/(rho*D1*q))^(1/4);
    P0 = -2.3599e9*q0^3 -1.8082e7*q0^2 +4.3346e6*q0 + 9.4355e4;
    
    pin = pbh - rho*g*h1 - F1;  % Pintake
    H = CH*H0*(fq/f0)^2;        % Head
    P = Cp*P0*(fq/f0)^3;        % Potencia

    ypk = [pin; H];

end