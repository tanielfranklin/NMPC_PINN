import tensorflow as tf
import numpy as np
from data.parameters_old import Parameters
import keras

class PinnPredictor(object):
    def __init__(self,path_model):
        #self.x0 = x0
        self.parameters=Parameters()
        self.path_model = path_model
        self.model=self.get_model()
    def get_model(self):
        print(f"Loading model from {self.path_model}")
        return keras.models.load_model(self.path_model)
        
    def norm_u(self,u):
        aux=[]
        u[2]=u[2]-self.parameters.u0[0] 
        u[3]=u[3]-self.parameters.u0[1]
        for i,valor in enumerate(u):
            aux.append(valor/self.parameters.uc[i])
        return np.hstack(aux)
    def norma_x(self,x):
        xn=[(x[:,i]-self.parameters.x0[i])/self.parameters.xc[i] for i in range(3)]
        return np.array(xn).T
    # def next_step(self,x):
    #     return self.model.predict(x)
    # def many_steps(self,tstart,nsim,Xdata,ydata,udata):
    #     X=Xdata[tstart:tstart+1,:,:]
    #     y=ydata[tstart:tstart+nsim+1,:,:]
    #     u=udata[tstart:tstart+nsim+1,:,:]
    #     U=Xdata[tstart:tstart+nsim+1,:,:4]
    #     Xx=X[:,:,-2:]
    #     X0=X
    #     y0=self.model.predict(X0)[:,0:1,:]
    #     pred=y0
    #     for i in range(nsim):
    #         Xx=tf.concat([Xx[:,1:,:],y0[:,:,:-1]],1) # Remove o instante mais antigo e atualiza o vetor de estados com a nova predição (remove q) 
    #         #print(U[i:i+1,:,:].shape)
    #         X0=tf.concat([U[i:i+1,:,:],Xx],2) # Remonta o vetor de entrada da rede (exógenas+saidas)
    #         y0=model.predict(X0)[:,0:1,:]
    #         pred=tf.concat([pred,y0[:,0:1,:]],0)
    #     return pred,y,u
