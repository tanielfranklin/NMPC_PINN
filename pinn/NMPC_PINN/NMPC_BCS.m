%% Nesse codigo, compilei a função fmincon em um executavel MEX, 
% assim o tempo computacional do solver reduz consideravelmente!!
% codegen -lang:c++ solverNMPC -args {uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,xmk,ymk,pm,DuYsp0,Ain,Bin,Aeq,beq,Dumin,Dumax} -report

%%
clear all
close all
clc
bcs_settings_hlim % executa arquivo do envelope op.

% steady-state conditions
xss = [8311024.82175957; 2990109.06207437; 0.00995042241351780; 50; 50];
uss = [50; 50];
yss = [6000142.88550200; 592.126490003812];
% Controller parameters
Hp = 10   ; % prediction horizon
Hc = 2    ; % control horizon
Ts = 2; % sampling time
% ----------------------------------------------
% q = 1*[1 1]; % weights on controlled variables 
% r = [0.1 0.1]; % weights on control actions
% qu = 1e5;
% ------Normalização através dos pesos----------
q = [1e6 1e8] ./(yss'.^2); % weights on controlled variables 
r = [10 1] ./(uss'.^2); % weights on control actions
qu = 1000 ./(uss(2).^2);
% -----------------------------------------------
ny=size(q,2);
nu=size(r,2);
%--------------------------------------------------------------------------
% constraints on inputs
%--------------------------------------------------------------------------
umin  = [35 0]'   ;  % lower bounds of inputs
umax  = [65 100]'   ;  % upper bounds of inputs 
dumax = Ts*[0.5 0.5]' ;  % maximum variation of input moves: [0.5 Hz/s; 0.5 %/s]
% Auxiliary matrices for using in the solver
Dumin = @(ymin) [repmat(-dumax,Hc,1);ymin];
Dumax = @(ymax) [repmat(dumax,Hc,1);ymax];

Mtil=[];
Itil=[];
auxM=zeros(nu,Hc*nu);
for in=1:Hc
    auxM=[eye(nu) auxM(:,1:(Hc-1)*nu)];
    Mtil=[Mtil;auxM];
    Itil=[Itil;eye(nu)];
end
% Ain = [Mtil;-Mtil];
Ain = [Mtil zeros(Hc*nu,ny),;-Mtil zeros(Hc*nu,ny)];
Bin = @(uk_1) [repmat(umax,Hc,1) - Itil*uk_1; Itil*uk_1 - repmat(umin,Hc,1)];
Aeq = [];
beq = [];
%-------------------- Linearizaçao para o EKF -----------------------------
[Ak,Bk,Ck,Dk] = linearizacao_bcs;   % Calculo das Jacobianas
% EKF
% Variancia da medição (R)
V = ((0.01/3)*diag(yss)).^2;
% Variancia do modelo (Q)
W = ((0.01/3)*diag(xss)).^2;
% Variancia da estimacao
Mk = W;
%
Vruido = ((0.01/3)*diag(yss)).^2;
%--------------------------------------------------------------------------
% Initial condition (steady-state preferentially)
%--------------------------------------------------------------------------
uk_1 = [50 50]'; % manipulated variable
x0 = [8311024.82175957;2990109.06207437;0.00995042241351780;50;50]; % state variables
ypk = yss;
xmk = x0; ymk = yss;  % condição inicial para o EKF
xmk(3) = 0.0106; % inicia a vazao de um x0 diferente para testar converg.
xmk2 = x0; ymk2 = yss; % estados da simulação do modelo nominal
utg = 90;   % target na choke
pm = 2e6;   % pressão da manifold
% Limites de pressao de intake
ymin(1,1) = yss(1); % Pressao de intake
ymax(1,1) = yss(1); % Pressao de intake
% Regiao de operação
hlim = BCS.Envelope.Hlim(x0(3)); 
ymin(2,1) = min(hlim); % Downtrhust
ymax(2,1) = max(hlim); % Uptrhust
%--------------------------------------------------------------------------
DuYsp0  = [zeros(Hc*nu,1); yss]; % Initial guess for decision variables (\Delta u_k)
%----------------Simulação----------------------------
tsim = 700;     % seconds 
nsim=tsim/Ts;   % number of steps

Xmk_all(:,1) = xmk;
Xk_all(:,1) = x0;

for k = 1:nsim
    k
    tsim = k*Ts;
%     changes on set-points Pintake
    switch tsim
        case Ts
            ymin(1) = 8.8e6;
        case 150
            ymin(1) = 6e6;
        case 400
            ymin(1) = 4.2e6;
        case 470
            pm = 8e5;
    end
    %ymin(1,1) = yss(1);    % Pressao de intake
    ymax(1,1) = ymin(1,1); % Pressao de intake
    
    %% Limite Up e Downthrust
    hlim = BCS.Envelope.Hlim(xmk(3));
    ymin(2) = min(hlim);
    ymax(2) = max(hlim);
%%   control law 

    tic
    [DuYsp,fval,flag,report] = solverNMPC_mex(uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,xmk,ymk,ymk2,pm,DuYsp0,Ain,Bin(uk_1),Aeq,beq,Dumin(ymin),Dumax(ymax));
%     [DuYsp,fval,flag,report] = solverNMPC(uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,xmk,ymk,ymk2,pm,DuYsp0,Ain,Bin(uk_1),Aeq,beq,Dumin(ymin),Dumax(ymax));
    tcalc(k)=toc;
    
    uk(:,k) = uk_1 + DuYsp(1:nu);
    J_k(k) = fval ; % cost function
    uk_1 = uk(:,k); % optimal input at time step k
    iter(1,k) = report.iterations;
    evalObj(1,k) = report.funcCount;
    flags(1,k) = flag;
    DuYsp0 = DuYsp; % Estimativa inicial do otmizador
        
%%  Plant
    tspan = [0 Ts];
    if k <= 80
        [t,xpk] = ode45(@(t,h)bcs_model_plant(t,h,[uk_1; pm]),tspan,x0);
    else
        [t,xpk] = ode45(@(t,h)bcs_model_plant(t,h,[uk_1; pm]),tspan,x0);
    end
    x0 = xpk(end,:);
    ypk = eq_medicao(x0);
    ruido = 5;
    ypk = ypk + ruido*mvnrnd(zeros(ny,1),Vruido)';  % ruido -> 3x a 5x
    yp(k,:) = ypk';
    Xk(:,k) = x0;
    % Potência
    y_sea = eq_algebricas_bcs(x0);
    Pk(k,:) = y_sea(end);
    Xk_all(:,k+1) = x0;

    %% EKF
    tic
    [xmk,Mk] = EKF_Bruno(Ak,Bk,Ck,xmk,ypk,[uk_1; pm],W,V,Mk,Ts);
    timeEKF(k) = toc;
    ymk = eq_medicao(xmk);  % saídas filtradas
    Ym(k,:) = ymk';
    Xmk(:,k) = xmk;
    Xmk_all(:,k+1) = xmk;
       
    %%  Nominal Model
    [t,xmk2] = ode45(@(t,h)bcs_model(t,h,uk_1),tspan, xmk2);
    xmk2 = xmk2(end,:);
    ymk2 = eq_medicao(xmk2);

%%  
    e(:,k) = ymk - ymk2; % erro de predição    
    sp(:,k) = DuYsp(nu*Hc+1:end); % Set-points
    Ymin(:,k) = ymin;
    Ymax(:,k) = ymax;
    
end
%% Plots
time = Ts:Ts:nsim*Ts;
figure(1)
label = {'p_{in}(bar)','H(m)'};
for iy = 1:ny
    subplot(ny,1,iy)
    hold on
    if iy == 1
        plot(time,yp(:,iy)/1e5,'.',time,yp(:,iy)/1e5,time,sp(iy,:)/1e5,'-.k')
    else
        plot(time,yp(:,iy),'.',time,sp(iy,:),'-.k',time,[Ymin(iy,:);Ymax(iy,:)],'--r')
    end
    grid on
    ylabel(label(iy))
end
xlabel('Time /(s)')
legend('Medição','Sp','Faixa')

label = {'f(Hz)','z_c(%)'};
figure(2)
for iu = 1:nu
    subplot(nu,1,iu)
    hold on
%     stairs(time,uk(iu,:))
    plot(time,uk(iu,:))
    plot([1,tsim],[umin(iu), umin(iu)],'--r')
    plot([1,tsim],[umax(iu), umax(iu)],'--r')
%     ylabel(label(iu))
    grid on
end
plot([1,tsim],[utg, utg],'-.k')
xlabel('Time /(s)')

figure(3)
hold on
plot(time,J_k)
grid on
xlabel('Time /(s)')
ylabel('Cost function')

figure(4)
hold on
bar(time,tcalc(1:end)*1000)
grid on
xlabel('\textbf{time (s)}', 'interpreter', 'latex')
ylabel('\textbf{Computational time (ms)}', 'interpreter', 'latex')

figure(5)
hold on
plot(Xk(3,:)*3600,yp(:,2),'-')
hold on
plot(Xk(3,1)*3600,yp(1,2),'o','MarkerFaceColor',[0,1,0],'MarkerEdgeColor',[0,0,0])
plot(Xk(3,end)*3600,yp(end,2),'o','MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
text(Xk(3,1)*3600,yp(1,2),'t = 0','HorizontalAlignment','left')
text(Xk(3,end)*3600,yp(end,2),sprintf('t = %d', tsim),'HorizontalAlignment','left')
BCS.Envelope.fig();
grid on
xlabel('q_p (m^3/h)')
ylabel('H (m)')
hold off

label = {'ep(Pa)','ep(m)'};
figure(6)
for iy = 1:ny
    subplot(ny,1,iy)
    hold on
    plot(time,e(iy,:))
    ylabel(label(iy))
    grid on
end
xlabel('Time /(s)')

figure
plot(flags)
% figure
% plot(tempo,Xk_all(3,:)*3600,'b','LineWidth',1.5)
% hold on
% plot(tempo,Xmk_all(3,:)*3600,'k-.','LineWidth',1.5)
% legend('\textbf{real flow}','\textbf{estimated flow}', 'interpreter', 'latex')
% xlabel('\textbf{time (s)}', 'interpreter', 'latex')
% ylabel('$\mathbf{q_p (m^3/h)}$', 'interpreter', 'latex')

% indx = trapz(time,J_k)
% producao = trapz(time,Xk(3,:)); % m3
% energia = trapz(time,Pk)/(3600*1000); %kWh