function [y] = open_loop_sim(x0m,du,uk_1,Hp,nu,ny,Ts,pm)
y = zeros(Hp*ny,1);
i = 2;
for k = 1:Hp
    uk_1 = uk_1 + du((k-1)*nu+1:k*nu) ;
    [t,xk] = ode45(@(t,h)bcs_model(t,h,[uk_1]),[0 Ts],x0m) ;
    x0m = xk(end,:)';    
    ymk = eq_medicao(x0m); % dimension ny  
    y(i-k:i-k+1,:) = ymk; % dimension Hp*ny
    i = i+3;
end

end

