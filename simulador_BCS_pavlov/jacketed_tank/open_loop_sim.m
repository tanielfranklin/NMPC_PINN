function [y] = open_loop_sim(x0m,du,uk_1,Hp,nu,ny,Ts,modelo)
y=[];
for k = 1:Hp
    uk_1 = uk_1 + du((k-1)*nu+1:k*nu) ;
    [t,ys] = ode45(@(t,h)jacketed_tank(t,h,modelo,uk_1),[0 Ts],x0m) ;
    ym(k,:) = ys(end,:); % dimension ny  
    x0m = ym(k,:)';
    y=[y;ym(k,:)']; % dimension Hp*ny
end

end

