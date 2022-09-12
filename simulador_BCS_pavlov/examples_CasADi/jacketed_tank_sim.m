clear all
close all
clc

modelo.rho = 1e3       ;
modelo.Cp  = 4.18      ;
modelo.At  = pi*0.5^2  ;
modelo.k   = 7         ;
modelo.lambda = 2.257e4;
modelo.Ti = 40 ;

% Estado estacionario na entrada
Fi = 10;
% Ti = 40;
Fv = 10;

% Estado estacionario no estados
L = 2.0408;
T = 45.4  ;
x0=[L;T];

Ts=0.1;
tsim=50;
nsim=tsim/Ts;
for j=1:nsim
    if j <= 2/Ts
        Fi=10; Fv=10; uk_1=[Fi;Fv];
    elseif j >2/Ts && j<=30/Ts
        Fi=15; Fv=10; uk_1=[Fi;Fv];
    else
       Fi=15; Fv=10*1.1; uk_1=[Fi;Fv];
    end
    tic
    [t,y]=ode45(@(t,x)jacketed_tank(t,x,modelo,uk_1),[0 Ts],x0);
    tcalc(j)=toc;
    yp(j,:)=y(end,:);
    x0=yp(j,:)';
end
figure(1)
subplot(2,1,1)
plot(Ts:Ts:nsim*Ts,yp(:,1))
xlabel('tempo /min')
ylabel('nivel /m')
grid on
subplot(2,1,2)
plot(Ts:Ts:nsim*Ts,yp(:,2))
xlabel('tempo /min')
ylabel('Temperature /ºC')
grid on

figure(2)
bar(1:nsim,tcalc,'k')
title('Time at each time step')
axis([1 nsim 0 0.05])