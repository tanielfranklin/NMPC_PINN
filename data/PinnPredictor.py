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
    
    
    def start_dataset(self,ui,xi):
        # Normalizar
        xi=self.parameters.normalizar_x(xi)
        #xi=[xi[0,0],xi[0,1]]
        ui=self.parameters.normalizar_u(ui)
        Xi=tf.convert_to_tensor(np.repeat([xi],20,axis=1), dtype=tf.float32)# Replicate to build NN input
        Ui=tf.convert_to_tensor(np.repeat([ui],20,axis=1), dtype=tf.float32) # Replicate to build NN input
        return Ui,Xi
    def update_inputs(self,y0,uk,Ui,Xi):
        uk=self.parameters.normalizar_u(uk)
        Xi=tf.concat([Xi[:,1:,:],y0[:,:,:-1]],1) # Remove older time-step and update states vector with new predictions (remove q)
        Ui=tf.concat([Ui[:,1:,:],np.array([uk])],1) # Remove older time-step and update exogenous vector with the next time-step
        return Ui,Xi
    
    def predict_pinn(self,Ui,Xi):
        y=self.model(tf.concat([Ui,Xi],2))
        yout=y[:,0,:].numpy()
        yout=self.parameters.desnorm_x(yout.T)
        return y,yout.T
        
    
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
