import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from data.LossHistory import LossHistory
from data.VarHistory import VarHistory
from data.TrainState import TrainState
import tensorflow_probability as tfp
from data.utils import dydt, get_abs_max_grad
from data.param import *
import numpy as np
import time

rho=950
PI=2.32e-9

## PINN elaborated by tanielfranlin@gmail.com
#Adapted from https://colab.research.google.com/drive/1lo7Kf8zTb-DF_MjkO8Y07sYELnX3BNUR#scrollTo=-rWEI708GDei 
#L-BFGS inspired in example Pi-Yueh Chuang <pychuang@gwu.edu>
class pinn_vfm(object):
    #Define the Constructor
    # def __init__(self,neurons, optimizer, logger,  var=None, batch_size=1, pinn_mode=1, inputs=2, error=None):
    def __init__(self,neurons, optimizer, logger,  var=None, pinn_mode=1, inputs=2, n_steps_in=20,n_steps_out=1,parameters=None):    # Descriptive Keras model LSTM model

        self.u_model = Sequential()
        #self.batch_size=batch_size
        self.ode_parameters=parameters
        
        self.inputs=inputs #input states
        # kernel_initializer='glorot_uniform',
        # bias_initializer=None,

        n_features=inputs


        # encoder layer
        self.u_model.add(LSTM(neurons, input_shape=(n_steps_in, inputs)))
        self.u_model.add(Dropout(0.2))
        self.u_model.add(RepeatVector(n_steps_out))
        # decoder layer
        self.u_model.add(LSTM(neurons, return_sequences=True))
        self.u_model.add(Dropout(0.2))  
        self.u_model.add(TimeDistributed(Dense(3)))#,input_shape=(n_steps_out, 2)))
        output_start=self.u_model.predict(tf.random.normal(shape=(1,n_steps_in,inputs),dtype=tf.float32)) # forçar o inicio dos parâmetros do modelo
        

        self.bestWeights=self.u_model.get_weights()
        
        self.bestLoss=np.inf
        self.optimizer = optimizer   
        self.logger = logger
        self.dtype = tf.float32
        
        #self.Loss_Weight_pinn=loss_weight_pinn
        self.alfa=tf.constant(0.8, dtype=tf.float32)
        self.lamb_bc=tf.constant(1.0, dtype=tf.float32)
        self.lamb_l1=tf.constant(1.0, dtype=tf.float32)
        self.lamb_l2=tf.constant(1.0, dtype=tf.float32)
        self.lamb_l3=tf.constant(1.0, dtype=tf.float32)
        self.pinn_mode=1
    
        self.pinn_mode_set(pinn_mode)

        self.loss_new=True
        if self.pinn_mode!=0:
            self.rho = tf.Variable(var[0], dtype=tf.float32)
            self.PI = tf.Variable(var[1], dtype=tf.float32)
        else:
            self.rho=tf.Variable(1.0, dtype=tf.float32)
            self.PI=tf.Variable(1.0, dtype=tf.float32)



        #self.error_fn=self.erro()
        self.Font=14   
        
        self.losshistory = LossHistory() 
        self.varhistory = VarHistory()
        self.train_state = TrainState()
        self.nsess=0 
        self.epoch=0
        self.logger.log_train_start(self)
        self.test_X=tf.random.normal(shape=(1,n_steps_in,inputs),dtype=tf.float32)
        self.test_y=tf.random.normal(shape=(1,n_steps_in,n_steps_out),dtype=tf.float32)
    
    def get_lamb_weights(self):   
        l1=f"[{self.lamb_bc.numpy():4.3f},{self.lamb_l1.numpy():4.3f},{self.lamb_l2.numpy():4.3f},{self.lamb_l3.numpy():4.3f}"
        #l2=f"{self.lamb_l1.numpy():4.3f},{self.lamb_l2.numpy():4.3f},{self.lamb_l3.numpy():4.3f}]"
        return l1


    def pinn_mode_set(self,value):
        cases = {
            # Turn off all loss terms linked with EDO
            "off": lambda: 0,
            # Turn off all loss terms linked with EDO
            "on": lambda: 1,
            # Turn on all loss terms linked with EDO
            "all": lambda: 2,
            # Turn off main loss_EDO and keep loss EDO2
            "loss2": lambda: 3,
        }
        if value in cases.keys():        
            self.pinn_mode=cases[value]()
        else:
            raise ValueError("Invalid arguments for pinn_mode")

    def function_factory(self, train_x, train_y,uk):
        #function used to L-BFGS Adapted from Pi-Yueh Chuang <pychuang@gwu.edu>
        # Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
        #
        # Distributed under terms of the MIT license.

        """An example of using tfp.optimizer.lbfgs_minimize to optimize a TensorFlow model.

        This code shows a naive way to wrap a tf.keras.Model and optimize it with the L-BFGS
        optimizer from TensorFlow Probability.

        Python interpreter version: 3.6.9
        TensorFlow version: 2.0.0
        TensorFlow Probability version: 0.8.0
        NumPy version: 1.17.2
        Matplotlib version: 3.1.1
        """
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.

        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.wrap_training_variables())
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        #@tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.
            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.wrap_training_variables()[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        #@tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
            params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                #loss_value = self.myloss(self.u_model(train_x, training=True), train_y)
                #loss_value = self.myloss(self.u_model(train_x), train_y)
                loss_bc,loss_x1,loss_x2,loss_x3,loss_f= self.GetLoss(train_y, self.u_model(train_x),uk)   
                loss_f=self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+self.lamb_l3*loss_x3
                #loss_f=loss_x1+loss_x2+loss_x3
                loss_value=self.lamb_bc*loss_bc+loss_f
  

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.wrap_training_variables())
            grads = tf.dynamic_stitch(idx, grads)
            

            # print out iteration & loss
            f.iter.assign_add(1)
            self.train_state.step=str(f.iter.numpy())
            self.train_state.rho=self.rho*rho
            self.train_state.PI=self.PI*PI
            self.train_state.loss_test=self.erro().numpy()
            self.train_state.loss_train=loss_value.numpy()
            self.train_state.loss_train_bc=self.lamb_bc*loss_bc.numpy()
            self.train_state.loss_train_f=loss_f.numpy()
            self.train_state.loss_train_x1=self.lamb_l1*loss_x1.numpy()
            self.train_state.loss_train_x2=self.lamb_l2*loss_x2.numpy()
            self.train_state.loss_train_x3=self.lamb_l3*loss_x3.numpy()
            self.train_state.weights=self.u_model.get_weights()
            self.train_state.update_best() 
            if f.iter%10==0:
                #tf.print("Iter:", f.iter, loss:{loss_value:.4e}")
                #tf.print("Iter:", f.iter, "loss:", loss_value)
                custom_log_res=f"[{loss_x1.numpy()*self.lamb_l1.numpy():.2e},{loss_x2.numpy()*self.lamb_l2.numpy():.2e},{loss_x3.numpy()*self.lamb_l3.numpy():.2e}]"
                custom_log=f"{self.rho.numpy()*rho:.1f} {self.PI.numpy()*PI*3.6e8:.4f} ({(100*np.abs(950-self.rho.numpy()*rho)/950):1.3f}%) ({(100*np.abs(PI-self.PI.numpy()*PI)/PI):1.3f}%)"
                #custom_log=f" lambda=[{self.lamb_l1:.1f},{self.lamb_l2:.1f},{self.lamb_l3:.1f}], rho={self.rho.numpy()*rho:.1f}, PI={self.PI.numpy()*PI:.2e}"
                self.logger.log_train_epoch(f.iter.numpy(), loss_value.numpy(),loss_f.numpy(), loss_bc.numpy(),custom=custom_log_res+custom_log)
                self.losshistory.append(
                    f.iter.numpy(),
                    self.train_state.loss_train,
                    self.train_state.loss_train_bc,
                    self.train_state.loss_train_f,
                    self.train_state.loss_train_x1,
                    self.train_state.loss_train_x2,
                    self.train_state.loss_train_x3,
                    self.train_state.loss_test,
                    None)
                self.varhistory.append(
                    f.iter.numpy(),
                    self.rho.numpy()*rho,
                    self.PI.numpy()*PI)
 
            # if f.iter%200==0:
            #     # #Updating adaptive lambda values
            #     self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
            #                                                               self.lamb_l2,
            #                                                               self.lamb_l3,
            #                                                               self.lamb_bc,
            #                                                               train_x,
            #                                                               train_y,
            #                                                               uk)
                
            #     # self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
            #     #                                                                 self.lamb_l2,
            #     #                                                                 self.lamb_l3,
            #     #                                                                 self.lamb_bc,
            #     #                                                                 x_batch_train,y_batch_train,u_batch)
            #     print(f"l1:{self.lamb_l1}, l2:{self.lamb_l2}, l3:{self.lamb_l3}, lbc:{self.lamb_bc}")

            # store loss value so we can retrieve later
            #tf.py_function(f.history.append, inp=[loss_value], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(self.epoch)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.history = []

        return f
    @tf.function
    def erro(self): 
        y_pred = self.u_model(self.test_X)
        yr=self.test_y
        #erro=tf.sqrt((yr[:,-1,:] - y_pred[:,-1,:])**2)
        erro=tf.square(yr[:,:,0:2] - y_pred[:,:,0:2])
        return tf.reduce_mean(erro)

    @tf.function
    def ED_BCS(self,x,u):
        var=[self.rho, self.PI]
        #init=time.time()
        ddy=dydt(x,tf.constant(1.0,dtype=tf.float32))
      
        # Tensores (Estados atuais preditos)
        # pbh = x[:,:,0:1]*pbc+pbmin
        # pwh = x[:,:,1:2]*pwc+pwmin
        # q = x[:,:,2:]*qc+qmin #Vazão
        # #Entradas exógenas atuais
        # ### A entrada da rede exige entradas normalizadas o cálculo dos resíduos não
        # fq=u[:,:,:1] *60  # desnormalizar para EDO
        # zc=u[:,:,1:2]*100 # desnormalizar para EDO
        # pm=u[:,:,2:3]*pm_c+pm0
        # pr=u[:,:,3:]*prc+pr0


        #Teste derivada
        # Tensores (Estados atuais preditos)
        pbh = x[:,0,0:1]*self.ode_parameters.pbc+self.ode_parameters.pbmin
        pwh = x[:,0,1:2]*self.ode_parameters.pwc+self.ode_parameters.pwmin
        q = x[:,0,2:]*self.ode_parameters.qc+self.ode_parameters.qmin #Vazão
        #Entradas exógenas atuais
        ### A entrada da rede exige entradas normalizadas o cálculo dos resíduos não
        fq=u[:,0,:1] *60  # desnormalizar para EDO
        zc=u[:,0,1:2]*100 # desnormalizar para EDO
        pm=u[:,0,2:3]*self.ode_parameters.pm_c+self.ode_parameters.pm0
        pr=u[:,0,3:]*self.ode_parameters.prc+self.ode_parameters.pr0

        # print(fq.shape,zc.shape,pm.shape,pr.shape)
        # print(pbh.shape,pwh.shape,q.shape)
        # print("ddy:",ddy.shape)




        # Calculo do HEAD e delta de press�o
        
        q0 = q / Cq * (f0 / fq)
        H0 = -1.2454e6 * q0 ** 2.0 + 7.4959e3 * q0 + 9.5970e2
        H = CH * H0 * (fq / f0) ** 2.0  # Head
        F1 = 0.158 * ((var[0]*rho* L1 * (q) ** 2.0) / (D1 * A1 ** 2.0)) * (mu / (var[0]*rho* D1 * (q))) ** (1.0/4.0)
        F2 = 0.158 * ((var[0]*rho * L2 * (q) ** 2.0) / (D2 * A2 ** 2.0)) * (mu / (var[0]*rho* D2 * (q))) ** (1.0/4.0)
        qr = var[1]*PI * (pr - pbh)
        qch = (zc/100.0)*Cc * tf.sqrt(tf.abs(pwh-pm));
        ##########################
        qch=(qch-self.ode_parameters.qch_lim[0])/self.ode_parameters.qcc
        F1=(F1-self.ode_parameters.F1lim[0])/self.ode_parameters.F1c
        F2=(F2-self.ode_parameters.F2lim[0])/self.ode_parameters.F2c
        H=(H-self.ode_parameters.H_lim[0])/self.ode_parameters.Hc

        dy1=- (1/self.ode_parameters.pbc)*b1/V1*(qr - q)
        dy2=- (1/self.ode_parameters.pwc)*b2/V2*(q - (self.ode_parameters.qcc*qch+self.ode_parameters.qch_lim[0]))
        dy3=- (1/(self.ode_parameters.qc*M))*(pbh - pwh - var[0]*rho*g*hw - (self.ode_parameters.F1c*F1+self.ode_parameters.F1lim[0])  - (self.ode_parameters.F2c*F2+self.ode_parameters.F2lim[0]) +  var[0]*rho* g * (H*self.ode_parameters.Hc+self.ode_parameters.H_lim[0]))
        #return tf.reduce_mean(tf.square(ddy[:,:,0:1]+dy1)), tf.reduce_mean(tf.square(ddy[:,:,1:2]+dy2)), tf.reduce_mean(tf.square(ddy[:,:,2:]+dy3))
        return tf.reduce_mean(tf.square(ddy[:,0:1]+dy1)), tf.reduce_mean(tf.square(ddy[:,1:2]+dy2)), tf.reduce_mean(tf.square(ddy[:,2:]+dy3))



    @tf.function    
    def GetLoss(self,y, y_pred,u): 
        #pinn_mode ="off" 0# Turn off all loss terms linked with EDO
        #     "on"1# Turn on the main loss term linked with EDO
        #     "all"2# Turn on all loss terms linked with EDO
        #    "loss2"3# Turn off main loss_EDO and keep new loss EDO
        #remove non measured variable from loss mse error
        ysliced=y_pred[:,:,0:2]
        loss_obs=tf.reduce_mean(tf.square(y - ysliced))
        if self.pinn_mode==1 or self.pinn_mode==2:
            #computing the residues with predicted states
            r1,r2,r3=self.ED_BCS(y_pred,u)
        else:
            r1,r2,r3=tf.constant(0.0, dtype=tf.float32),tf.constant(0.0, dtype=tf.float32),tf.constant(0.0, dtype=tf.float32)
        if self.pinn_mode==2 or self.pinn_mode==3:
            #Using measured Pbh and Pwh to compute the residues
            R1,R2,R3=self.ED_BCS(tf.concat([y[:,:,0:2],y_pred[:,:,2:]],axis=2),u)
        else:
            R1,R2,R3=tf.constant(0.0, dtype=tf.float32),tf.constant(0.0, dtype=tf.float32),tf.constant(0.0, dtype=tf.float32)
        return loss_obs,r1,r2,r3,(r1+r2+r3)

    #@tf.function
    def GetLamb(self,lamb_bc,X,y,u):
        
        with tf.GradientTape(persistent=True) as tape:
            lb,l1,l2,l3,lf=self.GetLoss(y, self.u_model(X),u)
            
        grad_f = tape.gradient(lf,  self.wrap_training_variables())
        grad_bc = tape.gradient(lb,  self.u_model.trainable_weights)
        del tape
        #print(get_abs_max_grad(grad_f),(1-alfa)*lamb_bc)
        lamb_bc=tf.convert_to_tensor( (1-self.alfa)*lamb_bc+self.alfa*get_abs_max_grad(grad_f)/get_abs_mean_grad(grad_bc),dtype=tf.float32)     
        
        return lamb_bc
    def GetLambStates(self,lamb_l1,lamb_l2,lamb_l3,lamb_bc,X,y,u):
        def update_lamb(l,gradmax,gradx):
            return tf.convert_to_tensor((1-pinn_vfm.alfa)*l
                                    +pinn_vfm.alfa*get_abs_max_grad(gradmax)/get_abs_mean_grad(gradx),dtype=tf.float32)
        try:
            with tf.GradientTape(persistent=True) as tape:
                lb,l1,l2,l3,lf,R1,R2,R3=self.GetLoss(y, self.u_model(X),u)  
                
            grad_f = tape.gradient(lf,  self.wrap_training_variables())           
            grad_l1 = tape.gradient(l1,  self.wrap_training_variables())
            grad_l2 = tape.gradient(l2,  self.wrap_training_variables())
            grad_l3 = tape.gradient(l3,  self.wrap_training_variables())
            grad_bc = tape.gradient(lb,  self.u_model.trainable_weights) #remember that it does't depend on rho and PI
            del tape
            print("Gradientes")
            print(f'grad_bc:{get_abs_max_grad(grad_bc):1.2e}, grad_1:{get_abs_max_grad(grad_l1):1.2e}, grad_2:{get_abs_max_grad(grad_l2):1.2e}, grad_3:{get_abs_max_grad(grad_l3):1.2e}')
            #print(get_abs_max_grad(grad_f),(1-alfa)*lamb_bc)
            ##### Fixo grad_bc atualiza l1,l2,l3 ############
            # lamb_l1=tf.convert_to_tensor( (1-self.alfa)*lamb_l1
            #                             +self.alfa*get_abs_max_grad(grad_bc)/get_abs_mean_grad(grad_l1),dtype=tf.float32)     
            # lamb_l2=tf.convert_to_tensor( (1-self.alfa)*lamb_l2
            #                             +self.alfa*get_abs_max_grad(grad_bc)/get_abs_mean_grad(grad_l2),dtype=tf.float32)     
            # lamb_l3=tf.convert_to_tensor( (1-self.alfa)*lamb_l3
            #                             +self.alfa*get_abs_max_grad(grad_bc)/get_abs_mean_grad(grad_l3),dtype=tf.float32)     
            
            ##### Fixo l3 atualiza l1,l2,grad_bc ############
                        #print(get_abs_max_grad(grad_f),(1-alfa)*lamb_bc)
            lamb_l1=update_lamb(lamb_l1,grad_l1,grad_l1)
            lamb_l2=update_lamb(lamb_l2,grad_l1,grad_l2)
            lamb_l3=update_lamb(lamb_l3,grad_l1,grad_l3)
            lamb_bc=update_lamb(lamb_bc,grad_l1,grad_bc) 
        except Exception:
            print(traceback.format_exc()) 
        return lamb_l1,lamb_l2,lamb_l3,lamb_bc
    

    def get_params(self):
        rho = self.rho
        PI = self.PI
        return rho

    #@tf.function
    def wrap_training_variables(self):
        var = self.u_model.trainable_weights
        if self.pinn_mode!=0:
            var.extend([self.rho])
            var.extend([self.PI])
        #var.extend([self.rho, self.PI])
        
        return  var


    @tf.function
    def GetGradAndLoss(self,y,X,u):
        with tf.GradientTape() as tape:
            #init=time.time()
            #print("Loss computing")
            loss_bc,loss_x1,loss_x2,loss_x3,loss_f = self.GetLoss(y, self.u_model(X),u)   
            #end = time.time()
            #print(f"Runtime computing losses {end - start}")
            
            
            #loss_value=self.lamb_bc*loss_bc+loss_f
            loss_value=self.lamb_bc*loss_bc+self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+self.lamb_l3*loss_x3
            #loss_value=loss_bc*self.lamb_bc+self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+loss_x3*self.lamb_l3
        loss_f=self.lamb_l1*loss_x1+self.lamb_l2*loss_x2+self.lamb_l3*loss_x3
        grads = tape.gradient(loss_value,  self.wrap_training_variables())
        return grads,loss_bc,self.lamb_l1*loss_x1,self.lamb_l2*loss_x2,self.lamb_l3*loss_x3,loss_value,loss_f
            
    
    def fit(self, train_data, tf_epochs=5000,adapt_w=False):
        train_dataset,self.test_X,self.test_y=train_data
        self.logger.set_error_fn(self.erro)
        if adapt_w==True:
            self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
                                                                            self.lamb_l2,
                                                                            self.lamb_l3,
                                                                            self.lamb_bc,
                                                                            train_X,train_y, u_train)
        self.logger.log_train_opt("Adam",self.pinn_mode,self.get_lamb_weights())
        self.logger.start_time=time.time()


        
        
        try:       
            for epoch in range(tf_epochs):
                # Iterate over the batches of the dataset.         
                for step, (x_batch_train, y_batch_train, u_batch) in enumerate(train_dataset):
                    #init=time.time()
                    grads,loss_bc,loss_x1,loss_x2,loss_x3,loss_value,loss_f=self.GetGradAndLoss(y_batch_train,x_batch_train,u_batch)
                    #print(f"Runtime of grad and loss {time.time() - init}")
                    if np.isnan(loss_value.numpy()):
                        print("Nan values appear. Stopping training",loss_x1.numpy(),loss_x2.numpy(),loss_x3.numpy(),loss_bc.numpy())
                        self.logger.log_train_end(tf_epochs,self.train_state)
                        self.summary_train(self.train_state)
                        raise Exception("Loss with Nan values found")
                    #print("save 1")
                    self.train_state.step=str(self.epoch)
                    # print("save 2")
                    self.train_state.rho=self.rho*rho
                    self.train_state.PI=self.PI*PI
                    self.train_state.loss_test=self.erro().numpy()
                    self.train_state.loss_train=loss_value
                    self.train_state.loss_train_bc=self.lamb_bc*loss_bc.numpy()
                    self.train_state.loss_train_f=loss_f.numpy()
                    self.train_state.loss_train_x1=self.lamb_l1*loss_x1.numpy()
                    self.train_state.loss_train_x2=self.lamb_l2*loss_x2.numpy()
                    self.train_state.loss_train_x3=self.lamb_l3*loss_x3.numpy()
                    self.train_state.weights=self.u_model.get_weights()
                    self.train_state.update_best()
                    
                    self.optimizer.apply_gradients(zip(grads, self.wrap_training_variables()))
                    # end = time.time()
                    # print(f"Runtime of batch  {end - start}")
                if (epoch%300==0 and adapt_w==True):
                    #init=time.time()
                    #self.lamb_l1,self.lamb_l2,self.lamb_l3=self.GetLambStates(self.lamb_l1,self.lamb_l2,self.lamb_l3,x_batch_train,y_batch_train,u_batch)
                    self.lamb_l1,self.lamb_l2,self.lamb_l3,self.lamb_bc=self.GetLambStates(self.lamb_l1,
                                                                                           self.lamb_l2,
                                                                                           self.lamb_l3,
                                                                                           self.lamb_bc,
                                                                                           x_batch_train,y_batch_train,u_batch)
                    print(f'==============Weights===============')
                    print(f'[ bc ,  r1 ,  r2 ,  r3 ]')
                    print(f'{self.get_lamb_weights()}')
                #     self.lamb_bc=self.GetLamb(self.lamb_bc,x_batch_train,y_batch_train,u_batch)   
                #     #print(f"Runtime of update_lamb {time.time() - init}")

                if epoch%20==0:    
                    self.losshistory.append(
                        self.epoch ,
                        self.train_state.loss_train.numpy(),
                        self.train_state.loss_train_bc.numpy(),
                        self.train_state.loss_train_f,
                        self.train_state.loss_train_x1.numpy(),
                        self.train_state.loss_train_x2.numpy(),
                        self.train_state.loss_train_x3.numpy(),
                        self.train_state.loss_test,                        None)
                    self.varhistory.append(
                        self.epoch ,
                        self.rho.numpy()*rho,
                        self.PI.numpy()*PI)     

                custom_log_res=f"|{loss_x1.numpy()*self.lamb_l1.numpy():.2e},{loss_x2.numpy()*self.lamb_l2.numpy():.2e},{loss_x3.numpy()*self.lamb_l3.numpy():.2e}|"
                custom_log=f"{self.rho.numpy()*rho:.1f} {self.PI.numpy()*PI:.2e}"
                self.logger.log_train_epoch(self.epoch, loss_value, loss_f, loss_bc, custom=custom_log_res+custom_log)
                # if self.epoch==100:
                #     #self.optimizer.learning_rate=self.optimizer.learning_rate/2
                #     self.optimizer.learning_rate=0.01
                # if self.epoch==500:
                #         #self.optimizer.learning_rate=self.optimizer.learning_rate/2
                #         self.optimizer.learning_rate=0.001
                self.epoch=self.epoch+1
                self.nsess=self.nsess+tf_epochs # Save epochs for the next session
                self.varhistory.update_best(self.train_state)
            self.logger.log_train_end(tf_epochs,self.train_state)
            self.summary_train(self.train_state)
        
        except Exception as err:
            print(err.args)
            raise      

        return self.losshistory, self.train_state, self.varhistory 


        

    def fit_LBFGS(self, dataset, nt_config):
        train_x,train_y,u_train,self.test_X,self.test_y=dataset
        self.logger.set_error_fn(self.erro)
        
        #self.logger.log_train_start(self)
        
        self.logger.log_train_opt("LBFGS",self.pinn_mode,self.get_lamb_weights())
        self.logger.start_time=time.time()
        func=self.function_factory(train_x, train_y,u_train)
        init_params = tf.dynamic_stitch(func.idx, self.wrap_training_variables())
        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            num_correction_pairs=nt_config.nCorrection,
            tolerance=nt_config.tol,
            parallel_iterations=nt_config.parallel_iter,
            max_iterations=nt_config.maxIter,
            f_relative_tolerance=nt_config.tolFun
            )
        
        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        # do some prediction
        #pred_outs = self.u_model(x_batch_train)
        #err = np.abs(pred_outs[:,0:-1]-y_batch_train)
        #print("L2-error norm: {}".format(np.linalg.norm(err)/np.sqrt(11)))
        print("Converged:",results.converged.numpy())
        print("Didn't find a step to satisfy:",results.failed.numpy())
        print("Exausted evaluations:",True if (results.converged.numpy() and results.failed.numpy())==False else False)
        print("Nb evals:",results.num_objective_evaluations.numpy())
        

        self.epoch=func.iter.numpy()
        self.logger.log_train_end(self.epoch,self.train_state)
        self.summary_train(self.train_state)
        return self.losshistory, self.train_state, self.varhistory

    def use_best_weights(self):
        self.u_model.set_weights(self.train_state.best_test_weights)
    def disregard_best_weights(self):
        self.train_state.best_weights=np.inf



    def summary_train(self, train_state):
        print(f"Best model at step:{train_state.best_test_step}, Best rho:{train_state.best_test_rho:.1f}, Best PI:{train_state.best_test_PI:.4e}\
         ({(100*np.abs(950-self.rho.numpy()*rho)/950):1.3f}%) ({(100*np.abs(PI-self.PI.numpy()*PI)/PI):1.3f}%)")
        print("  train loss: {:.2e}".format(train_state.best_test_train))
        print("  test loss: {:.2e}".format(train_state.best_test_loss))
        #print("  test metric: {:s}".format(list_to_str(train_state.best_metrics)))
        # if train_state.best_ystd is not None:
        #     print("  Uncertainty:")
        #     print("    l2: {:g}".format(np.linalg.norm(train_state.best_ystd)))
        #     print(
        #         "    l_infinity: {:g}".format(
        #             np.linalg.norm(train_state.best_ystd, ord=np.inf)
        #         )
        #     )
        #     print(
        #         "    max uncertainty location:",
        #         train_state.X_test[np.argmax(train_state.best_ystd)],
        #     )
    # print("")
        self.is_header_print = False

        #self.logger.log_train_end(tf_epochs + nt_config.maxIter)
    def summary_model(self):
        return self.u_model.summary()
    
           
