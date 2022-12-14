
from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns

class PlotResult(object):
    def __init__(self,Ts):
        self.label=["$P_{bh}(bar)$","$P_{wh}(bar)$", "$q (m^3/h)$", "f(Hz)",r"$z_c$(%)"]
        self.sc=np.array([1/1e5, 1/1e5,3600,1, 1],dtype=float)    
        self.label_y=["$P_{in}(bar)$","$H(m)$"]
        self.sc_y =np.array([1/1e5, 1],dtype=float)
        self.Ts=Ts
        
            
    def plot_resultado(self,X, U):
            obs=X
            
            u_test=U
            Font=14
            k=np.arange(0,obs.shape[1])
            Fig=plt.figure(figsize=(5, 5))
            sc=self.sc       
            for i,lb in enumerate(self.label):        
                ax1=Fig.add_subplot(len(self.label),1,i+1)
                # ax1.plot(k, obs[:,i]*sc[i],"-k", label='Valor esperado')
                y=obs[i,:]*sc[i]

                ax1.plot(k/(60/self.Ts),y,"-k", label='Valor esperado')
                # ax1.plot(k, pred_test[:,i]*sc[i],":",color='blue',lw=2,label='Predição')
                ax1.set_ylabel(lb,  fontsize=Font)
                if i!=len(self.label)-1:
                    ax1.set_xticklabels([])
                if i==0:
                    plt.legend()            
                ax1.grid(True)
            ax1.set_xlabel('$Tempo (min)$' ,  fontsize=Font)
            #plt.legend(bbox_to_anchor=(1, 3.8), ncol = 3)
            return Fig
    def plot_y(self,ysp, y,hlim,pin_lim):
        
        Font=14
        k=np.arange(0,ysp.shape[1])
        Fig=plt.figure(figsize=(5, 5))
        sc=self.sc_y      
        for i,lb in enumerate(self.label_y):        
            ax1=Fig.add_subplot(len(self.label_y),1,i+1)
            ax1.plot(k/(60/self.Ts),ysp[i,:]*sc[i],":b", label='Set point')
            ax1.plot(k/(60/self.Ts),y[i,:]*sc[i],"-k", label='Medição')
            if i==0:
                ax1.plot(k/(60/self.Ts),pin_lim[1,:]*sc[i],":r", label='Bound')
                ax1.plot(k/(60/self.Ts),pin_lim[0,:]*sc[i],":r")
            else:
                ax1.plot(k/(60/self.Ts),hlim[1,:],":r")
                ax1.plot(k/(60/self.Ts),hlim[0,:],":r")
            # ax1.plot(k, pred_test[:,i]*sc[i],":",color='blue',lw=2,label='Predição')
            ax1.set_ylabel(lb,  fontsize=Font)
            if i!=len(self.label_y)-1:
                ax1.set_xticklabels([])
            if i==0:
                plt.legend()            
            ax1.grid(True)
        ax1.set_xlabel('$Tempo (min)$' ,  fontsize=Font)
        #plt.legend(bbox_to_anchor=(1, 3.8), ncol = 3)
        return Fig
    
    def plot_pinn_model(self,df_pinn,df_mod):
        df_mod['q']=df_mod['q']*3600
        df_pinn['q']=df_pinn['q']*3600
        df_mod['pbh']=df_mod['pbh']/1e5
        df_pinn['pbh']=df_pinn['pbh']/1e5
        df_mod['pwh']=df_mod['pwh']/1e5
        df_pinn['pwh']=df_pinn['pwh']/1e5

        # print(df_pinn.head())
        # print(df_mod.head())
        fig_res, axes = plt.subplots(3, 1)
        sns.lineplot(data=df_pinn, x='t', y='pbh', ax=axes[0], legend="full")
        sns.lineplot(data=df_mod, x='t', y='pbh', ax=axes[0],legend="full")
        g1=sns.lineplot(data=df_pinn, x='t', y='pwh', ax=axes[1])
        sns.lineplot(data=df_pinn, x='t', y='q', ax=axes[2])
        g2=sns.lineplot(data=df_mod, x='t', y='pwh', ax=axes[1])
        sns.lineplot(data=df_mod, x='t', y='q', ax=axes[2])
        axes[0].set(xlabel=None)
        axes[1].set(xlabel=None)
        axes[0].set_xticklabels([])
        axes[1].set_xticklabels([])
        g1.set_label("Pinn")#  legend(label="pinn")
        g2.set_label("Phenomena")
        #print(g.set_label["text"])
        g1.legend(loc='upper left',labels=["Pinn",'',"Model"])
        return fig_res



