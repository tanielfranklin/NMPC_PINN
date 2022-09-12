function J = fob_NMPC(du,uk_1,Hp,Hc,ysp,q,r,Ts,x0m,x0,nu,ny,modelo)

xv = [du;zeros(nu*(Hp-Hc),1)]; % du: decision variables Hc*nu
ym = open_loop_sim(x0m,xv,uk_1,Hp,nu,ny,Ts,modelo); % dimension Hp*ny

Q = diag(repmat(q,1,Hp)); % dimension Hp*ny x Hp*ny 
R = diag(repmat(r,1,Hc)); % dimension Hc*nu x Hc*nu
ysp = repmat(ysp,Hp,1)  ; % dimension Hp*ny

% Bias 
e=x0-x0m; % mismatch at time step k
ee=repmat(e,Hp,1);
y=ym+ee; % augmented vector of prediction plus bias

% Objective function of the controller
J = (y-ysp)'*Q*(y-ysp) + du'*R*du;
end
