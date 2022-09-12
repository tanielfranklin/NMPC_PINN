H0_dt = -1.2454e6*q0_dt**2 + 7.4959e3*q0_dt + 9.5970e2;
H0_dt = CH*H0_dt*(f0/f0)**2;
H0_ut = -1.2454e6*q0_ut**2 + 7.4959e3*q0_ut + 9.5970e2;
H0_ut = CH*H0_ut*(f0/f0)**2;

f0=60
f = np.linspace(30,70,1000); #% Hz
H_ut = H0_ut*(f/f0)**2;
H_dt = H0_dt*(f/f0)**2;

Qdt = q0_dt*f/f0;
Qut = q0_ut*f/f0;



flim = np.arange(35,70,5);
qop = np.linspace(0,q0_ut*flim[-1]/f0,1000); # m3/s
#qop=np.transpose(qop);


Hop = np.zeros((len(flim),len(qop)));


for i in range(0,len(flim)):
    q0 = qop/Cq*(f0/flim[i]);
    H0 = -1.2454e6*q0**2 + 7.4959e3*q0 + 9.5970e2;
    Hop[i,:] = CH*H0*(flim[i]/f0)**2;
    #print(i)
    


from shapely.geometry import LineString,Point,MultiPoint

# Calculo dos pontos de interse��o para delimita��o da regi�o
#points1=np.zeros((1000,1));
points1=[];points2=[];points4=[];points6=[];#points8=[];

for ind in range(0,999):
    points1.append((qop[ind]*3600,Hop[0,ind]));
    points2.append((Qdt[ind]*3600,H_dt[ind]));#2 e 3 são iguais
    #points3.append((Qdt[ind]*3600,H_dt[ind]));#2 e 3 são iguais
    points4.append((qop[ind]*3600,Hop[-1,ind])); # igual a 5
    #points5.append((qop[ind]*3600,Hop[-1,ind]));
    points6.append((Qut[ind]*3600,H_ut[ind]));
    #points7.append((Qut[ind]*3600,H_ut[ind])); #Igual ao 6
    #points8.append((qop[ind]*3600,Hop[1,ind])); #Igual ao 1
    
line1=LineString(points1); 
line2=LineString(points2); #igual a line3
line4=LineString(points4);
line6=LineString(points6);


ip=np.zeros((4,2));    
[ip[0,0],ip[0,1]] = [line2.intersection(line1).x, line2.intersection(line1).y];
[ip[1,0],ip[1,1]] = [line4.intersection(line2).x, line4.intersection(line2).y];
[ip[2,0],ip[2,1]] = [line6.intersection(line4).x, line6.intersection(line4).y];
[ip[3,0],ip[3,1]] = [line6.intersection(line1).x, line6.intersection(line1).y];
                  

# Ajuste do polinomio de frequencia maxima 65 Hz
p_35hz = np.polyfit(qop*3600,Hop[0,:],3);
H_35hz = lambda qk: p_35hz@np.vstack((cumprod_reverse(np.tile(qk,(len(p_35hz)-1,1)),0),np.ones((1,len(qk)))));
q_35hz = np.linspace(ip[0,0],ip[3,0],100);
# Ajuste do polinomio de frequencia minima 35 Hz
p_65hz = np.polyfit(qop*3600,Hop[-1,:],3);
H_65hz = lambda qk: p_65hz@np.vstack((cumprod_reverse(np.tile(qk,(len(p_65hz)-1,1)),0),np.ones((1,len(qk)))));
q_65hz = np.linspace(ip[1,0],ip[2,0],100);
# Ajuste do polinomio de Downtrhust
p_dt = np.polyfit(Qdt*3600,H_dt,2);
H_dt = lambda qk: p_dt@np.vstack((cumprod_reverse(np.tile(qk,(len(p_dt)-1,1)),0),np.ones((1,len(qk)))));
q_dt = np.linspace(ip[0,0],ip[1,0],100);
# Ajuste do polinomio de Uptrhust
p_ut = np.polyfit(Qut*3600,H_ut,2);
H_ut = lambda qk: p_ut@np.vstack((cumprod_reverse(np.tile(qk,(len(p_ut)-1,1)),0),np.ones((1,len(qk)))));
q_ut = np.linspace(ip[3,0],ip[2,0],100);

# % Constu��o da figura
# BCS.Envelope.fig = @(aux) plot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2);
# BCS.Envelope.ip = ip;
# BCS.Envelope.fBounds = struct('H_35hz',H_35hz,'H_65hz',H_65hz,'H_dt',H_dt,'H_ut',H_ut);
# % Funa��o para a avalia��o dos limites dada uma vaz�o.
# BCS.Envelope.Hlim = @(qk) BoundHead(qk*3600,ip,BCS.Envelope.fBounds);


def grafico_envelope(ax):
    ax.plot(q_35hz,H_35hz(q_35hz),':r'); 
    ax.plot(q_65hz,H_65hz(q_65hz),':r');
    ax.plot(q_ut,H_ut(q_ut),':r');
    ax.plot(q_dt,H_dt(q_dt),':r');
    ax.set_xlabel(r'$q_p (m^3/h)$')
    ax.set_ylabel('H (m)')

#% Constu��o da figura
#figBCS=lambda aux: pyplot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2)
fBounds = {'H_35hz': H_35hz,
           'H_65hz': H_65hz,
           'H_dt': H_dt,
           'H_ut': H_ut}
BCS['Envelope'] = {'fig': grafico_envelope, #lambda aux: plt.plot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2),
                   'ip': ip,
                   'fbounds': fBounds}

    
    

# % Funa��o para a avalia��o dos limites dada uma vaz�o.
BCS['Envelope']['Hlim']= lambda qk: BoundHead(qk,ip,BCS['Envelope']['fBounds']);
#


#%% Subrotina
def BoundHead(qk,ip,bounds):
    if qk < ip[0,0]:
        Hlim = [ip[0,1],ip[0,1]];
    elif qk < ip[1,0]:
        Hlim = [[bounds['H_35hz'][qk]],[bounds['H_dt'][qk]]];
    elif qk < ip[3,0]:
        Hlim = [[bounds['H_35hz'][qk]],[bounds['H_65hz'][qk]]];
    elif qk < ip[2,0]:
        Hlim = [[bounds['H_ut'][qk]],[[bounds['H_65hz'][qk]]]];
    else:
        Hlim = [ip[2,1],ip[2,1]];
    
    return Hlim

def cumprod_reverse (A,n):
    #Como não havia cumprod reverse no python tivemos que contornar
    if n==0:
        return np.flipud(np.cumprod(np.flipud(A),n));
    elif n==1:
        return np.fliplr(np.cumprod(np.fliplr(A),n));
    else:
        print('Erro em n - cumprod reverse')

print('Envelope carregado')