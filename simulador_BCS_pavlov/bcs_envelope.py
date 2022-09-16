
import pandas as pd
import seaborn as sns
from shapely.geometry import LineString
from param import *
from matplotlib import pyplot as plt 

def cumprod_reverse (A,n):
    #Como não havia cumprod reverse no python tivemos que contornar
    if n==0:
        return np.flipud(np.cumprod(np.flipud(A),n));
    elif n==1:
        return np.fliplr(np.cumprod(np.fliplr(A),n));
    else:
        print('Erro em n - cumprod reverse')

class BcsEnvelope(object):
    def __init__(self):
        f0=60
        H0_dt = -1.2454e6*q0_dt**2 + 7.4959e3*q0_dt + 9.5970e2
        H0_dt = CH*H0_dt*(f0/f0)**2;
        H0_ut = -1.2454e6*q0_ut**2 + 7.4959e3*q0_ut + 9.5970e2
        H0_ut = CH*H0_ut*(f0/f0)**2;

        f = np.linspace(30,70,1000); ## Hz
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
        [ip[0,0],ip[0,1]] = [line2.intersection(line1).x, line2.intersection(line1).y]
        [ip[1,0],ip[1,1]] = [line4.intersection(line2).x, line4.intersection(line2).y]
        [ip[2,0],ip[2,1]] = [line6.intersection(line4).x, line6.intersection(line4).y]
        [ip[3,0],ip[3,1]] = [line6.intersection(line1).x, line6.intersection(line1).y]
        self.ip=ip        

        # Ajuste do polinomio de frequencia maxima 65 Hz
        p_35hz = np.polyfit(qop*3600,Hop[0,:],3);
        self.H_35hz = lambda qk: p_35hz@np.vstack((cumprod_reverse(np.tile(qk,(p_35hz.shape[0]-1,1)),0),np.ones((1,(1)))))
        self.H_35hz_env = lambda qk: p_35hz@np.vstack((cumprod_reverse(np.tile(qk,(p_35hz.shape[0]-1,1)),0),np.ones((1,len(qk)))))

        self.q_35hz = np.linspace(ip[0,0],ip[3,0],100);
        # Ajuste do polinomio de frequencia minima 35 Hz
        p_65hz = np.polyfit(qop*3600,Hop[-1,:],3);
        self.H_65hz = lambda qk: p_65hz@np.vstack((cumprod_reverse(np.tile(qk,((p_65hz.shape[0])-1,1)),0),np.ones((1,(1)))));
        self.H_65hz_env = lambda qk: p_65hz@np.vstack((cumprod_reverse(np.tile(qk,((p_65hz.shape[0])-1,1)),0),np.ones((1,len(qk)))));
        
        self.q_65hz = np.linspace(ip[1,0],ip[2,0],100);
        # Ajuste do polinomio de Downtrhust
        p_dt = np.polyfit(Qdt*3600,H_dt,2);
        self.H_dt = lambda qk: p_dt@np.vstack((cumprod_reverse(np.tile(qk,((p_dt.shape[0])-1,1)),0),np.ones((1,(1)))));
        self.H_dt_env = lambda qk: p_dt@np.vstack((cumprod_reverse(np.tile(qk,((p_dt.shape[0])-1,1)),0),np.ones((1,len(qk)))));

        self.q_dt = np.linspace(ip[0,0],ip[1,0],100);
        # Ajuste do polinomio de Uptrhust
        p_ut = np.polyfit(Qut*3600,H_ut,2);
        self.H_ut = lambda qk: p_ut@np.vstack((cumprod_reverse(np.tile(qk,((p_ut.shape[0])-1,1)),0),np.ones((1,(1)))));
        self.H_ut_env = lambda qk: p_ut@np.vstack((cumprod_reverse(np.tile(qk,((p_ut.shape[0])-1,1)),0),np.ones((1,len(qk)))));

        self.q_ut = np.linspace(ip[3,0],ip[2,0],100);

        # # Constu��o da figura
        # self.Envelope.fig = @(aux) plot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2);
        # self.Envelope.ip = ip;
        # self.Envelope.fBounds = struct('H_35hz',H_35hz,'H_65hz',H_65hz,'H_dt',H_dt,'H_ut',H_ut);
        # # Funa��o para a avalia��o dos limites dada uma vaz�o.
        # self.Envelope.Hlim = @(qk) BoundHead(qk*3600,ip,self.Envelope.fBounds);

        self.fBounds = {'H_35hz': self.H_35hz,
                'H_65hz': self.H_65hz,
                'H_dt': self.H_dt,
                'H_ut': self.H_ut}
        self.Hlim=lambda qk: self._BoundHead(qk)
        self.size_env=(8,8)
    def _BoundHead(self,qk):
        ip,bounds=self.ip,self.fBounds
        
        if (qk < ip[0,0]):
            
            Hlim = [ip[0,1],ip[0,1]];
        elif qk < ip[1,0]:
            Hlim = [[bounds['H_35hz'][qk]],[bounds['H_dt'][qk]]];
        elif qk < ip[3,0]:
            Hlim = [[bounds['H_35hz'][qk]],[bounds['H_65hz'][qk]]];
        elif qk < ip[2,0]:
            Hlim = [bounds['H_ut'](qk),bounds['H_65hz'](qk)];

        else:
            Hlim = [ip[2,1],ip[2,1]];
        #print(ip)
        return Hlim
    def grafico_envelope(self,Xk,Yk):
        fig=plt.figure(figsize=self.size_env)
        ax=fig.add_subplot(111)
        ax.plot(self.q_35hz,self.H_35hz_env(self.q_35hz),':b'); 
        ax.plot(self.q_65hz,self.H_65hz_env(self.q_65hz),':b');
        ax.plot(self.q_ut,self.H_ut_env(self.q_ut),':r');
        ax.plot(self.q_dt,self.H_dt_env(self.q_dt),':r');
        ax.set_xlabel(r'$q_p (m^3/h)$')
        ax.set_ylabel('H (m)')
        plt.plot(Xk[2,0:].T*3600,Yk[1,0:].T,'--k')
        plt.plot(Xk[2,0]*3600,Yk[1,0],'o')#,'MarkerFaceColor',[0,1,0],'MarkerEdgeColor',[0,0,0])
        plt.plot(Xk[2,-1]*3600,Yk[1,-1],'o')#,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
        plt.annotate('t=0',
                    xy=(float(Xk[2,0]*3600),float(Yk[1,0])),
                    xytext=(float(Xk[2,0]*3600)-5,float(Yk[1,0])+10),
                    arrowprops=dict(facecolor='green', shrink=0.01))

        plt.annotate(f't={Xk.shape[1]}',
                    xy=(float(Xk[2,-1]*3600),float(Yk[1,-1])),
                    xytext=(float(Xk[2,-1]*3600)-7,float(Yk[1,-1])+10),
                    arrowprops=dict(facecolor='red', shrink=0.01))
        return fig,ax
    def grafico_envelope2(self):
        fig=plt.figure(figsize=self.size_env)
        ax=fig.add_subplot(111)
        ax.plot(self.q_35hz,self.H_35hz_env(self.q_35hz),':b'); 
        ax.plot(self.q_65hz,self.H_65hz_env(self.q_65hz),':b');
        ax.plot(self.q_ut,self.H_ut_env(self.q_ut),':r');
        ax.plot(self.q_dt,self.H_dt_env(self.q_dt),':r');
        ax.set_xlabel(r'$q_p (m^3/h)$')
        ax.set_ylabel('H (m)')
        # plt.plot(Xk[2,0:].T*3600,Yk[1,0:].T,'--k')
        # plt.plot(Xk[2,0]*3600,Yk[1,0],'o')#,'MarkerFaceColor',[0,1,0],'MarkerEdgeColor',[0,0,0])
        # plt.plot(Xk[2,-1]*3600,Yk[1,-1],'o')#,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
        # plt.annotate('t=0',
        #             xy=(float(Xk[2,0]*3600),float(Yk[1,0])),
        #             xytext=(float(Xk[2,0]*3600)-5,float(Yk[1,0])+10),
        #             arrowprops=dict(facecolor='green', shrink=0.01))

        # plt.annotate(f't={Xk.shape[1]}',
        #             xy=(float(Xk[2,-1]*3600),float(Yk[1,-1])),
        #             xytext=(float(Xk[2,-1]*3600)-7,float(Yk[1,-1])+10),
        #             arrowprops=dict(facecolor='red', shrink=0.01))
        return fig,ax
    
    def grafico_envelope3(self):
        df = pd.DataFrame(dict(q35=self.q_35hz,
                H35=self.H_35hz_env(self.q_35hz)))
        g = sns.relplot(x="q35", y="H35", kind="line", data=df)
        #g = sns.relplot(x="q35", y="H65", kind="line", data=df)
        g.figure.autofmt_xdate()
        return g