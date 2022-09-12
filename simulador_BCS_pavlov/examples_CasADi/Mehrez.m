clear all
clc
addpath('C:\Users\marcio\Downloads\casadi-windows-matlabR2016a-v3.5.1')
import casadi.*

T = 0.2;  
N = 20  ;
rob_diam = 0.3;

v_max = 0.6; v_min = -v_max;
omega_max = pi/4; omega_min=-omega_max;

x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta');
states = [x;y;theta]; n_states=length(states);
v = SX.sym('v'); omega = SX.sym('omega');
controls = [v;omega]; n_controls=length(controls);
rhs = [v*cos(theta); v*sin(theta);omega];
f = Function('f',{states,controls},{rhs});

U = SX.sym('U',n_controls,N)      ; % decision variables (control action sequence)
P = SX.sym('P',n_states+n_states) ; % parameters (time-varying e.g initial state, uk_1, set-point, state mismatch)
X = SX.sym('X',n_states,N+1)      ; % predicted state trajectory over prediction horizon

X(:,1)=P(1:n_states);
for k=1:N
    st=X(:,k); con=U(:,k);
    f_value = f(st,con);
    st_next = st+T*f_value; % dx/dt = f(x,u) => (x_next-x)/T=f(x,u)
    X(:,k+1)=st_next; 
end

obj = 0;
q=[1 5 0.1]; Q=diag(q);
r=[0.5 0.05]; R=diag(r);

for k=1:N
    st=X(:,k); con=U(:,k);
    obj=obj+(st-P(4:6))'*Q*(st-P(4:6))+con'*R*con;
end

g = [];
for k=1:N+1
    g=[g;X(1,k)];
    g=[g;X(2,k)];
end

ff=Function('ff',{U,P},{X});

opt_variables = [U(:)]; %reshape(U,2*N,1); %reshape(X,3*(N+1),1)];
nlp_prob = struct('f',obj,'x',opt_variables, 'g', g, 'p', P);
options=struct;
options.ipopt.max_iter=100;
solver = nlpsol('solver', 'ipopt', nlp_prob, options);
args=struct;
args.lbg = -2;
args.ubg =  2;
args.lbx(1:2:2*N,1) = v_min;
args.ubx(1:2:2*N,1) = v_max;
args.lbx(2:2:2*N,1)   = omega_min;
args.ubx(2:2:2*N,1)   = omega_max;
t0 = 0;
x0 = [0;0;0];
xs = [1.5; 1.5; 0];
xx(:,1) = x0;
t(1)=t0;
u0 = zeros(N,2);
sim_tim = 20;
mpciter=0;
xx1 = [];
u_cl = [];

while norm(x0-xs)>1e-2 && mpciter < sim_tim/T
    args.p = [x0;xs];
    args.x0 = u0(:); 
    sol=solver('x0',args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    u = reshape(full(sol.x)',2,N)';
    ff_value = ff(u',args.p);
    xx1(:,1:3,mpciter+1)=full(ff_value)';
    u_cl=[u_cl ; u(1,:)];
    
    t(mpciter+1)=t0;
    [t0,x0,u0]=shfit(T,t0,x0,u,f);
    xx(:,mpciter+2)=x0;
    mpciter=mpciter+1;
end

figure(1)
plot(xx(1,:),xx(2,:))
figure(2)
subplot(2,1,1)
stairs(t,u_cl(:,1))
xlabel('tempo /s')
% ylabel('nivel /m')
grid on
subplot(2,1,2)
stairs(t,u_cl(:,2))
xlabel('tempo /s')
% ylabel('Temperature /�C')
grid on
% x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z');
% nlp = struct('x',[x;y;z], 'f',x^2+100*z^2, 'g',z+(1-x)^2-y);
% S = nlpsol('S', 'ipopt', nlp);
% disp(S)
% r = S('x0',[2.5,3.0,0.75],...
%       'lbg',0,'ubg',0);
% x_opt = r.x;
% disp(x_opt)