function J = fob_NMPC(DuYsp,uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,x0m,ymk,ymk2,pm)
du = DuYsp(1:nu*Hc);
ysp = DuYsp(nu*Hc+1:end);

xv = [du;zeros(nu*(Hp-Hc),1)]; % du: decision variables Hc*nu
ym = open_loop_sim(x0m,xv,uk_1,Hp,nu,ny,Ts,pm); % dimension Hp*ny

Qu = diag(qu);
Q = diag(repmat(q,1,Hp)); % dimension Hp*ny x Hp*ny 
R = diag(repmat(r,1,Hc)); % dimension Hc*nu x Hc*nu
ysp = repmat(ysp,Hp,1)  ; % dimension Hp*ny

% Bias 
e=ymk-ymk2; % mismatch at time step k
ee=repmat(e,Hp,1);
y=ym+ee; % augmented vector of prediction plus bias

for k = 1:Hc
uk_1 = uk_1 + du((k-1)*nu+1:k*nu);
end

% Objective function of the controller
J = (y-ysp)'*Q*(y-ysp) + du'*R*du + (uk_1(2) - utg)'*Qu*(uk_1(2) - utg);
end
