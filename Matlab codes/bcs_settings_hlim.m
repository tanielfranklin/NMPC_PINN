% Constantes
f0 = 60;
mu  = 0.025;  % Viscosity [Pa*s]
q0_dt = 25/3600; % Downtrhust flow at f0
q0_ut = 50/3600; % Uptrhust flow at f0
CH = -0.03*mu + 1;
Cq = 2.7944*mu^4 - 6.8104*mu^3 + 6.0032*mu^2 - 2.6266*mu + 1;
Cp = -4.4376*mu^4 + 11.091*mu^3 -9.9306*mu^2 + 3.9042*mu + 1;

%% Região de operação Downtrhust e Upthrust
H0_dt = -1.2454e6*q0_dt.^2 + 7.4959e3*q0_dt + 9.5970e2;
H0_dt = CH*H0_dt*(f0/f0).^2;
H0_ut = -1.2454e6*q0_ut.^2 + 7.4959e3*q0_ut + 9.5970e2;
H0_ut = CH*H0_ut*(f0/f0).^2;
% Variacao frequencia
f = linspace(30,70,1000); % Hz
H_ut = H0_ut*(f./f0).^2;
H_dt = H0_dt*(f./f0).^2;
% corrige lei da afinidade
Qdt = q0_dt.*f/f0;
Qut = q0_ut.*f/f0;
% Variacao vazao
flim = 35:5:65;
qop = linspace(0,q0_ut*flim(end)/f0,1000); % m3/s
Hop = zeros(length(flim),length(qop));
for i = 1:length(flim)
    q0 = qop./Cq*(f0/flim(i));
    H0 = -1.2454e6*q0.^2 + 7.4959e3*q0 + 9.5970e2;
    Hop(i,:) = CH*H0*(flim(i)/f0).^2;
end
% Calculo dos pontos de interseção para delimitação da região
[ip(1,1),ip(1,2)] = polyxpoly(qop*3600,Hop(1,:),Qdt*3600,H_dt);
[ip(2,1),ip(2,2)] = polyxpoly(Qdt*3600,H_dt,qop*3600,Hop(end,:));
[ip(3,1),ip(3,2)] = polyxpoly(qop*3600,Hop(end,:),Qut*3600,H_ut);
[ip(4,1),ip(4,2)] = polyxpoly(Qut*3600,H_ut,qop*3600,Hop(1,:));

% Ajuste do polinomio de frequencia maxima 65 Hz
p_35hz = polyfit(qop*3600,Hop(1,:),3);
H_35hz = @(qk) p_35hz*[cumprod(repmat(qk,length(p_35hz)-1,1),1,'reverse');ones(1,length(qk))];
q_35hz = linspace(ip(1,1),ip(4,1),100);
% Ajuste do polinomio de frequencia minima 35 Hz
p_65hz = polyfit(qop*3600,Hop(end,:),3);
H_65hz = @(qk) p_65hz*[cumprod(repmat(qk,length(p_65hz)-1,1),1,'reverse');ones(1,length(qk))];
q_65hz = linspace(ip(2,1),ip(3,1),100);
% Ajuste do polinomio de Downtrhust
p_dt = polyfit(Qdt*3600,H_dt,2);
H_dt = @(qk) p_dt*[cumprod(repmat(qk,length(p_dt)-1,1),1,'reverse');ones(1,length(qk))];
q_dt = linspace(ip(1,1),ip(2,1),100);
% Ajuste do polinomio de Uptrhust
p_ut = polyfit(Qut*3600,H_ut,2);
H_ut = @(qk) p_ut*[cumprod(repmat(qk,length(p_ut)-1,1),1,'reverse');ones(1,length(qk))];
q_ut = linspace(ip(4,1),ip(3,1),100);
% Constução da figura
BCS.Envelope.fig = @(aux) plot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2);
BCS.Envelope.ip = ip;
BCS.Envelope.fBounds = struct('H_35hz',H_35hz,'H_65hz',H_65hz,'H_dt',H_dt,'H_ut',H_ut);
% Função para a avaliação dos limites dada uma vazão.
BCS.Envelope.Hlim = @(qk) BoundHead(qk*3600,ip,BCS.Envelope.fBounds);

%% Subrotina
function Hlim = BoundHead(qk,ip,bounds)
    if qk < ip(1,1)
        Hlim = [ip(1,2),ip(1,2)];
    elseif qk < ip(2,1)
        Hlim = [bounds.H_35hz(qk);bounds.H_dt(qk)];
    elseif qk < ip(4,1)
        Hlim = [bounds.H_35hz(qk);bounds.H_65hz(qk)];
    elseif qk < ip(3,1)
        Hlim = [bounds.H_ut(qk);bounds.H_65hz(qk)];
    else
        Hlim = [ip(3,2),ip(3,2)];
    end
end