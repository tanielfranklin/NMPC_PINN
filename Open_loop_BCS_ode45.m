% close all
clear
clc

x0 = [8311024.82175957;2990109.06207437;0.00995042241351780;50;50];
uk_1 = [50 50 2e6]';
Ts = 1;

tsim = 200;
nsim = tsim/Ts;
for j=1:nsim
    
    tsim = j*Ts;
    if tsim <= 20
        uk_1 = [35 100 2e6]';
    elseif tsim >20 && tsim<=40
%         uk_1 = [60 50 2e6]';
    else
        uk_1 = [65 100 2e6]';
    end

   
    [t,y]=ode45(@(t,x)bcs_model(t,x,uk_1),[0 Ts],x0);
    xpk(j,:) = y(end,:);
    x0 = xpk(j,:);
    ypk = eq_medicao(x0);
    pin(j,:) = ypk(1);
    H(j,:) = ypk(2);
    uk(j,:) = uk_1;
    % Potência
    y_sea = eq_algebricas_bcs(x0);
    Pk(j,:) = y_sea(end);
    
end

%% Grafico
t = Ts:Ts:nsim*Ts;
figure(1)
label = {'p_{bh} /Pa','p_{wh} /Pa','q_{p}'};
for iy = 1:3
    subplot(3,1,iy)
    hold on
    plot(t,xpk(:,iy)) 
    ylabel(label(iy))
end
xlabel('time \(s)')

figure(2)
hold on
plot(t,pin(:,1)/1e5)
ylabel('p_{in} /bar');
xlabel('time \(s)')

figure(3)
hold on
plot(t,H(:,1))
ylabel('H /m');
xlabel('time \(s)')

label = {'f(Hz)','z_c(%)'};
figure(4)
for iu = 1:2
    subplot(2,1,iu)
    hold on
    stairs(t,uk(:,iu))
    ylabel(label(iu))
    grid on
end


