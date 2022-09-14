function [DuYsp,fval,flag,report] = solverNMPC(uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,xmk,ymk,ymk2,pm,DuYsp0,Ain,Bin,Aeq,beq,Dumin,Dumax)
options = optimoptions(@fmincon,'display','off','algorithm','sqp','MaxIterations',30);

[DuYsp,fval,flag,report] = fmincon(@(DuYsp)fob_NMPC(DuYsp,uk_1,Hp,Hc,q,r,qu,utg,Ts,nu,ny,ypk,xmk,ymk,ymk2,pm),DuYsp0,Ain,Bin,Aeq,beq,Dumin,Dumax,[],options);


end