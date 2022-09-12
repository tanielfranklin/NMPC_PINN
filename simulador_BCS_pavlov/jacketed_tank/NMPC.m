clear all
close all
clc

% Model parameters
modelo.rho = 1e3       ;
modelo.Cp  = 4.18      ;
modelo.At  = pi*0.5^2  ;
modelo.k   = 7         ;
modelo.lambda = 2.257e4;
modelo.Ti = 40 ;

% Plant model parameters
modelo_p.rho = 1e3       ;
modelo_p.Cp  = 4.18      ;
modelo_p.At  = pi*0.5^2  ;
modelo_p.k   = 7         ;
modelo_p.lambda = 2.257e4;
modelo_p.Ti = 40 ;

% Controller parameters
nsim = 100 ; % number of steps
Hp = 10   ; % prediction horizon
Hc = 2   ; % control horizon
q = 10*[1 1] ; % weights on controlled variables 
r = [0.1 0.1]  ; % weights on control actions
Ts = 0.1; % sampling time
ny=size(q,2);
nu=size(r,2);

%--------------------------------------------------------------------------
% constraints on inputs
%--------------------------------------------------------------------------
umin  = [5 5]'   ;  % lower bounds of inputs
umax  = [20 30]'   ;  % upper bounds of inputs 
dumax = [1 2]' ;  % maximum variation of input moves
% Auxiliary matrices for using in the solver
Dumax = dumax;
for i = 1:Hc-1
  Dumax = [Dumax;dumax];
end
Dumin = -Dumax;
Mtil=[];
Itil=[];
auxM=zeros(nu,Hc*nu);
for in=1:Hc
    auxM=[eye(nu) auxM(:,1:(Hc-1)*nu)];
    Mtil=[Mtil;auxM];
    Itil=[Itil;eye(nu)];
end
Ain = [Mtil;-Mtil];
Bin = @(uk_1) [repmat(umax,Hc,1) - Itil*uk_1; Itil*uk_1 - repmat(umin,Hc,1)];
Aeq = [];
beq = [];
%--------------------------------------------------------------------------
% Initial condition (steady-state preferentially)
%--------------------------------------------------------------------------
uk_1 = [10 10]'; % manipulated variable
x0 = [2.0408 45.4]'; % state variables
x0m = [2.0408 45.4]' ;
%--------------------------------------------------------------------------
du0  = zeros(Hc*nu,1); % Initial guess for decision variables (\Delta u_k)
for k = 1:nsim
%     k
%     changes on set-points
    if k <= 300
        ysp = [3 50]';
    else
        ysp = [3 50]';
    end
%   control law 
%     options = optimoptions(@fmincon,'display','off','algorithm','active-set');
%     options = optimoptions(@fmincon,'display','off','algorithm','sqp');
    options = optimoptions(@fmincon,'display','off','algorithm','interior-point');
    tic
    [du,fval] = fmincon(@(du)fob_NMPC(du,uk_1,Hp,Hc,ysp,q,r,Ts,x0m,x0,nu,ny,modelo),du0,Ain,Bin(uk_1),Aeq,beq,Dumin,Dumax,[],options);
    tcalc(k)=toc;
    uk(:,k) = uk_1 + du(1:nu);
    J_k(k) = fval ; % cost function
    uk_1 = uk(:,k); % optimal input at time step k
        
    % Plant
%     gPRoms
    tspan = [0 Ts];
    if k <= 80
        [t,ys] = ode45(@(t,h)jacketed_tank_plant(t,h,modelo_p,uk_1),tspan,x0);
    else
        [t,ys] = ode45(@(t,h)jacketed_tank_plant(t,h,modelo_p,1.1*uk_1),tspan,x0);
    end
    yp(k,:) = ys(end,:)+0*mvnrnd(zeros(1,ny),diag([.01 .01]));  
    x0 = yp(k,:)';
    
    % Model
%     ANN
    [t,ys2] = ode45(@(t,h)jacketed_tank(t,h,modelo,uk_1),tspan, x0m);
    ym(k,:) = ys2(end,:); %+ mvnrnd(zeros(ny,1),V);  
    x0m = ym(k,:)';
    e(:,k)=x0-x0m;
    % Estimativa inicial do otmizador-------------------------------------------   
     du0 = [du(nu+1:end);zeros(nu,1)]; 
     sp(:,k) = ysp;
end

figure(1)
for in = 1:ny
    subplot(ny,1,in)
    plot(Ts:Ts:nsim*Ts,yp(:,in),'k',Ts:Ts:nsim*Ts,sp(in,:),'r--')
    ylabel(['y_' num2str(in)])
    grid on
end
xlabel('Time /(min)')
% legend('Setpoint','Medicao','Modelo','Location','Best')

figure(2)
for iu = 1:nu
    subplot(nu,1,iu)
    stairs(Ts:Ts:nsim*Ts,uk(iu,:),'k')
    ylabel(['u_' num2str(iu)])
    grid on
end
xlabel('Time /(min)')

figure(3)
plot(Ts:Ts:nsim*Ts,J_k,'k')
xlabel('Time /(min)')
ylabel('Cost function')

figure(4)
plot(Ts:Ts:nsim*Ts,e,'k')
xlabel('Time /(min)')
ylabel('Erro')

figure(5)
bar(1:nsim,tcalc,'k')
title('Time at each time step')
axis([1 nsim 0 inf])
% ylabel('Erro')