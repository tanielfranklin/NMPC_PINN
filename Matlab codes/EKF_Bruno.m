function [xmk,Mk] = EKF_Bruno(A,B,C,xmk,ypk,uk_1,W,V,Mk,Ts)
    
    % EKF - Predicao
    [t,xpk] = ode45(@(t,h)bcs_model_plant(t,h,[uk_1]),[0 Ts],xmk);
    xmk = xpk(end,:)';
    ymk = eq_medicao(xmk);
    
%     Phi = eye(length(xmk)) + full(Ak)*Ts + (full(Ak)^2)*(Ts^2)/2 + (full(Ak)^3)*(Ts^3)/factorial(3); % Discretização
    
    % Linearizacao 
    x01 = xmk(1); x02 = xmk(2); x03 = xmk(3); x04 = xmk(4); x05 = xmk(5);
    uk1 = uk_1(1); uk2 = uk_1(2); uk3 = uk_1(3);
    A = eval(A);
    B = eval(B);
    C = eval(C);
    C = (C([1,2],:)); % separando a matiz C só para as pv (pin,H)
    % Discretização
    sys = ss(A,B(:,[1 2]),C,[]);
    sys_d = c2d(sys,Ts,'zoh');
    [Ak,Bk,Ck,~] = ssdata(sys_d);
    
    Phi = Ak;
    % Calculo da matriz de covariancia Mk
    Mk = Phi*Mk*Phi' + W;
    
    % EKF - correcao dos estados estimados
    Kf = Mk*Ck'/(Ck*Mk*Ck' + V);        % calculdo ganho
    Mk = (eye(length(xmk)) - Kf*Ck)*Mk; % atualizacao da matriz de variancia dos estados estimados
    xmk = xmk + Kf*(ypk - ymk);          % correcao dos estados

end

