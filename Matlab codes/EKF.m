function [xmk,ymk,Kf,Mk] = EKF(process,xmk,yk,uk_1,W,V,Mk,ts,pv)
    
    %     % Atualização da matriz A
    [Ak,Bk,Ck,Dk] = process.Linearizacao(xmk,uk_1);
    Ck = full(Ck(pv,:)); % separando a matiz C só para as pv

    % EKF - Predicao
    [xpk,ypk] = process.Simulation(xmk,uk_1);
    ymk = full(ypk);
    xmk = full(xpk);

    % Discretização
    sys = ss(full(Ak),full(Bk(:,[1 2])),full(Ck),[]);
    sys_d = c2d(sys,ts,'zoh');
%   sysd = ss(Ak,Bk,Ck,Dk,ts);
    [Ad,Bd,Cd,~] = ssdata(sys_d);
    
    % Calculo da matriz de covariancia Mk
    Phi = eye(length(xmk)) + full(Ak)*ts + (full(Ak)^2)*(ts^2)/2 + (full(Ak)^3)*(ts^3)/factorial(3); % Discretização
    
%     Phi = Ad;
    Mk = Phi*Mk*Phi' + W;
    
    % Linearizacao a cada k Atualização da matriz C
%     [~,~,Ck] = process.Linearizacao(xmk,uk_1);
%     Ck = full(Ck(pv,:)); % separando a matiz C só para as pv
    
    % EKF - correcao dos estados estimados
    Kf = Mk*Ck'/(Ck*Mk*Ck' + V);        % calculdo ganho
    Mk = (eye(length(xmk)) - Kf*Ck)*Mk; % atualizacao da matriz de variancia dos estados estimados
    xmk = xmk + Kf*(yk - ymk);          % correcao do estado

    [~,ypk] = process.Simulation(xmk,uk_1);
    ymk = full(ypk);
    
end

