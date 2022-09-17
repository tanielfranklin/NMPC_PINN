% set(findall(gcf,'-property','FontSize'),'FontSize',18);
close all
clc 
t = 0:Ts:(nsim*Ts);

% figure(1)
% label = {'p_{in}(bar)','H(m)'};
% for iy = 1:npv
%     subplot(npv,1,iy)
%     hold on
%     if iy == 1
%         plot(time,Yk(iy,:)/1e5,'.',time,Ymk(iy,:)/1e5,time,Ys(iy,:)/1e5,'-.k')
%     else
%         plot(time,Yk(iy,:),'.',time,Ymk(iy,:),time,Ys(iy,:),'-.k',time,[Ymin(iy,:);Ymax(iy,:)],'--g')
%     end
%     ylabel(label(iy))
% end
% xlabel('Time (nT)')
% legend('Medição','EKF','Sp','Faixa')

figure(1)
t = 0:Ts:(nsim*Ts);
hold on
plot(t,Ys(1,:)/1e5,'-.k')
plot(t,Yk(1,:)/1e5,'-','LineWidth',1.5)
ylabel('$\mathbf{p_{in} \;(bar)}$', 'interpreter', 'latex')
xlabel('\textbf{time (s)}', 'interpreter', 'latex')
legend('\textbf{setpoint}','\textbf{target OFF}','\textbf{target ON}', 'interpreter', 'latex')
legend('\textbf{measurements}','\textbf{filtered signal (EKF)}','\textbf{setpoint}', 'interpreter', 'latex')

figure(2)
t = 0:ts:(nsim*ts);
hold on
plot(t,Yk(2,:),'-')
plot(t,Ys(iy,:),'-.k')
plot(t,[Ymin(iy,:);Ymax(iy,:)],'--r','LineWidth',1.5)
ylabel('\textbf{H (m)}', 'interpreter', 'latex')
xlabel('\textbf{time (s)}', 'interpreter', 'latex')
legend('\textbf{target OFF}','\textbf{target ON}','\textbf{setpoint}','\textbf{zone}', 'interpreter', 'latex')

figure(2)
% t = 0:ts:(nsim*ts);
label = {'\textbf{f (Hz)}','$\mathbf{z_c\;(\%)}$'};
for iu = 1:nmv
    subplot(nmv,1,iu)
    hold on
    grid on
    box on
    plot(t,Uk(iu,:),'LineWidth',1.5)
    plot([1,nsim*ts],[umin(iu), umin(iu)],'--r','LineWidth',1.5)
    plot([1,nsim*ts],[umax(iu), umax(iu)],'--r','LineWidth',1.5)
    ylabel(label(iu), 'interpreter', 'latex')
end
xlabel('\textbf{time (s)}', 'interpreter', 'latex')
legend('\textbf{target OFF}','\textbf{target ON}','\textbf{constraints}', 'interpreter', 'latex')


figure(4)
hold on
plot(Xk(3,:)*3600,Yk(2,:),'LineWidth',1.5)
hold on
plot(Xk(3,1)*3600,Yk(2,1),'o','MarkerFaceColor',[0,1,0],'MarkerEdgeColor',[0,0,0])
plot(Xk(3,end)*3600,Yk(2,end),'o','MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
text(Xk(3,1)*3600,Yk(2,1),'t = 0','HorizontalAlignment','left')
text(Xk(3,end)*3600,Yk(2,end),sprintf('t = %d', nsim),'HorizontalAlignment','left')
BCS.Envelope.fig();
xlabel('$\mathbf{q_p (m^3/h)}$', 'interpreter', 'latex')
ylabel('\textbf{H (m)}', 'interpreter', 'latex')
legend('\textbf{target OFF}','\textbf{target ON}', 'interpreter', 'latex')

legend('\textbf{measurements}','\textbf{filtered signal (EKF)}', '\textbf{setpoint}','interpreter', 'latex')

figure(4)
hold on
% histogram(telap)
bar(telap)
xlabel('Time solver (s)')
ylabel('Frequency')
fprintf('   Tempo: %0.2f s \n',t)

figure(5)
hold on
plot(Fval,'LineWidth',1.5)
xlabel('Time (nT)')
ylabel('Cost function')

figure(6)
label = {'Erro /(bar)','Erro /(m)'};
for iy = 1:npv
    subplot(nmv,1,iy)
    hold on
    if iy == 1
        plot(time,Ep(iy,:)/1e5,'LineWidth',1.5)
    else
        plot(time,Ep(iy,:),'LineWidth',1.5)
    end
    ylabel(label(iy))
end
hold off
xlabel('Time (nT)')

figure(7)
t = 0:ts:(nsim*ts);
hold on
plot(t,Xpk(3,:)*3600,'LineWidth',1.5)
xlabel('time (s)')
ylabel('q_p (m^3/h)')
legend('target_{OFF}','target_{ON}')

figure(8)
hold on
plot(Dek/1e5,'LineWidth',1.5)
xlabel('Time (nT)')
ylabel('Pressão manifold (bar)')

% t = 0:ts:(nsim*ts)-1;
% producao = trapz(t,Qck); % m3
% energia = trapz(t,Pk)/(3600*1000); %kWh

%% Selecionar legendas especificas
% f=get(gca,'Children');
% legend([f(1),f(4),f(2)],'setpoint','measurement','constraints')