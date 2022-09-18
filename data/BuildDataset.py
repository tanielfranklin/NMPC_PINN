from data.utils import add_noise_norm, plot_u, plot_states_BCS
import tensorflow as tf
import numpy as np
import pandas as pd
from data.parameters_old import Parameters


def normalizar_x(x,xc,x0):
    xn=[(x[:,i]-x0[i])/xc[i] for i in range(3)]
    return np.array(xn).T

def split_data(n_in,n_out,data):
    a,b,c=split_sequences(data, n_in, n_out)
    x=a[:,:,:]
    y=b[:,:,-3:]
    u_train=b[:,:,0:4] #catch 0 1 and 2 values
    return x,y,u_train

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    #https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
 X, y, u = list(), list(),list()
 for i in range(len(sequences)):
  # find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out-1
  # check if we are beyond the dataset
  if out_end_ix > len(sequences)-1:
   break
  # gather input and output parts of the pattern
  seq_x, seq_y, seq_u= sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, :],sequences[end_ix-1:out_end_ix, :]
  X.append(seq_x)
  y.append(seq_y)
  u.append(seq_u)
 return np.array(X), np.array(y), np.array(u)# choose a number of time steps #change this accordingly



class BuildingDataset(object):
    def __init__(self,n_steps_in, n_steps_out,dados,split_point,batch_size):
        self.n_steps_in, self.n_steps_out,self.batch_size,self.parameters =n_steps_in, n_steps_out,batch_size,Parameters()
        #test_y,test_X,train_X, train_y_full, u_train,
        self.split_point=split_point 
        self.tempo=dados['t']
        self.train_X,self.train_y,self.train_y_full, self.u_train=None,None,None,None
        self.pack_plot=None
        self.dataset_full=self.dataset(dados)
        self.dataset_full_noisy=self.dataset_noisy()
        self.x,self.u,self.un,self.xn=None,None,None,None
        self.train_dataset=None
        self.train=None
        self.pack=None
        self.test_y=None
        self.test_X=None   
        self.prepare()
        # self.figs=self.gen_fig()
    def dataset(self,dados):
        def reshape_data(dataset,length):
            dataset_new=[]
            for i in dataset:
                dataset_new.append(i.reshape([length,1]))
            return dataset_new
        fk=dados['U'][:,0:1]
        zc=dados['U'][:,1:2]
        x1=dados['x1']
        x2=dados['x2']
        x3=dados['x3']
        pmc=dados['U'][:,2:3]
        pr=dados['U'][:,3:4]
        tempo=dados['t']
        maxtime = fk.shape[0]
        nsim=maxtime
        dataset_full=[x1,x2,x3,fk,zc,pmc,pr,tempo]
        dataset_full=reshape_data(dataset_full,nsim)
        return dataset_full
        
    def dataset_noisy(self):
        #------------------------------------------------
        ### Inserindo ruido
        sigma=[0.01,0.01,0.01,0.005,0.001,0.01,0.01]
        dataset_full_noisy=[]
        for i,d in enumerate(self.dataset_full[0:-1]):
            dataset_full_noisy.append(add_noise_norm(d,sigma[i]))
        dataset_full_noisy.append(self.tempo)
        return dataset_full_noisy
        #----------------------------------------------------

        # x1,x2,x3,fk,zc,pmc,pr,tempo=dataset_full
        #------------------------------------------------
        # # Reducing dataset size
    def gen_fig(self):
        Figs={}            
        uplot=self.dataset_full[3:-1]
        Fig_u=plot_u(uplot)
        Fig_x=plot_states_BCS(self.x,self.tempo)
        uplot=[self.un[:,i] for i in range(4)]
        Fig_un=plot_u(uplot)
        Fig_xn=plot_states_BCS(self.xn,self.tempo,norm=True)
        Figs["un"]=Fig_un
        Figs["xn"]=Fig_xn
        Figs["u"]=Fig_u
        Figs["x"]=Fig_x
        return Figs
    def prepare(self):
        

        
        self.x=np.hstack(self.dataset_full_noisy[0:3])
        self.u=np.hstack(self.dataset_full_noisy[3:7])
        self.xn=self.parameters.normalizar_x(self.x)
        self.un=self.parameters.normalizar_u(self.u)
        # print("Limites das exógenas")
        # for i in self.dataset_full_noisy[3:7]:
        #     print(f"Max:{max(i)}, Min: {min(i)}")
        
        
        df = pd.DataFrame(np.hstack([self.un,self.xn]),columns=['fn','zn','pmn','prn','pbh','pwh','q'])
        df_u = pd.DataFrame(np.hstack([self.u]),columns=['f','z','pm','pr'])
        dset = df.values.astype(float)
        du_set = df_u.values.astype(float)
        #dset_test=df_test.values.astype(float)
        X,y,u_train=split_data(self.n_steps_in, self.n_steps_out,dset)
        
        train_X_full , train_y_full, u_train = X[:self.split_point, :] , y[:self.split_point, :], u_train[:self.split_point, :]
        test_X_full , test_y_full = X[self.split_point:, :] , y[self.split_point:, :]
        uk=dset[0:self.split_point,0:4]
        #Remove unmeasured variable q from training dataset
        train_y=train_y_full[:,:,0:2]
        train_X=train_X_full[:,:,:-1]
        self.test_y=test_y_full[:,:,0:3]
        self.test_X=test_X_full[:,:,:-1]
        uk=tf.convert_to_tensor(uk, dtype=tf.float32) # u(k) para ODE
        self.train_X=tf.convert_to_tensor(train_X, dtype=tf.float32) # X(k) para ODE
        self.train_y=tf.convert_to_tensor(train_y, dtype=tf.float32) # y(k) para ODE
        self.train_y_full=tf.convert_to_tensor(train_y_full, dtype=tf.float32) # y(k) para ODE
        self.u_train=tf.convert_to_tensor(u_train, dtype=tf.float32) # y(k) para ODE
        self.pack_plot=[train_X, train_y,self.test_X,self.test_y]

        self.pack=y, train_y_full, u_train
    def adam_dataset(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_X,self.train_y, self.u_train))
        train_dataset = train_dataset.batch(self.batch_size)
        return [train_dataset,self.test_X,self.test_y] 
    def lbfgs_dataset(self):
        return [self.train_X,self.train_y,self.u_train,self.test_X,self.test_y]

        

