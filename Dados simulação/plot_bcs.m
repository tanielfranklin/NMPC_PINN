figure(1)
label = {'p_{in}(bar)','H(m)'};
for iy = 1:ny
    subplot(ny,1,iy)
    hold on
    if iy == 1
        plot(time,sp(iy,:)/1e5,'-.k',time,yp(:,iy)/1e5,'.')
    else
        plot(time,yp(:,iy),'.',time,sp(iy,:),'-.k',time,[Ymin(iy,:);Ymax(iy,:)],'--r')
    end
    grid on
    ylabel(label(iy))
end
xlabel('Time (s)')
legend('Medição','Sp','Faixa')

label = {'f(Hz)','z_c(%)'};
figure(2)
for iu = 1:nu
    subplot(nu,1,iu)
    hold on
%     stairs(time,uk(iu,:),'LineWidth',1.5)
    plot(time,uk(iu,:),'LineWidth',1.5)
    plot([1,tsim],[umin(iu), umin(iu)],'--r','LineWidth',1.5)
    plot([1,tsim],[umax(iu), umax(iu)],'--r','LineWidth',1.5)
    ylabel(label(iu))
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
bar(time,tcalc(1:end))
grid on
xlabel('Time /(kT)')
ylabel('Computation time /(s)')

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

