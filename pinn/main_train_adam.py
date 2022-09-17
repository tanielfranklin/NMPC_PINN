

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from data.pinn_BCS import pinn_vfm

from data.Logger import Logger
from data.utils import Struct, save_model_files
from data.TrainingReport import TrainingReport

#time = np.linspace(0, maxtime, 200) # Regular points inside the domain

with open("dataset01.pk", 'rb') as open_file:
        ds = pickle.load(open_file)

#========================================
# # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_config = Struct()
#Positive integer. The number of iterations allowed to run in parallel. 
nt_config.parallel_iter=2
#The maximum number of iterations for L-BFGS updates. 
nt_config.maxIter = 400
#Specifies the maximum number of (position_delta, gradient_delta) correction pairs to keep as implicit approximation of the Hessian matrix. 
nt_config.nCorrection = 50
#If the relative change in the objective value between one iteration and the next is smaller than this value, the algorithm is stopped. 
nt_config.tolfun=1e-5
#Specifies the gradient tolerance for the procedure. If the supremum norm of the gradient vector is below this number, the algorithm is stopped. 
nt_config.tol = 1e-5 #Specifies the gradient tolerance for the procedure. If the supremum norm of the gradient vector is below this number, the algorithm is stopped. 

##---------------------------------------

#========================================
# Creating the model and training
logger = Logger(frequency=100)
#logger.set_error_fn(error)
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
neurons=15
rho=950
PI=2.32*1e-9
start_rho=0.9*950/rho
start_PI=0.9*2.32*1e-9/PI
var=[start_rho, start_PI] # normalized parameters
n_features=6 # Network inputs  (fk, zc,pmc,pr, x1,x2)
nt_config.maxIter = 100
Nc=10
pinn = pinn_vfm(Nc,tf_optimizer, logger,
                var=var,pinn_mode="on", 
                inputs=n_features, 
                n_steps_in=ds.n_steps_in,
                n_steps_out=ds.n_steps_out,
                parameters=ds.parameters)

#######################################
pinn.lamb_l1=tf.constant(1.0, dtype=tf.float32) #x1 residue weight
pinn.lamb_l2=tf.constant(1.0, dtype=tf.float32) #x3 residue weight
pinn.lamb_l3=tf.constant(1.0, dtype=tf.float32) #x3 residue weight
# #######################################
dataset_adam=ds.adam_dataset()
loss_history, trainstate,var_history=pinn.fit(dataset_adam, tf_epochs=200)#,adapt_w=True)  
training_report=TrainingReport(pinn,[loss_history,trainstate,var_history],ds)
plt.show() # Uncomment to see the graphics


#Uncomment the lines below to save the model
folder_string="model_adam_200"
objects2save={"Loss":loss_history,"trainstate":trainstate,"vartrain":var_history}
save_model_files(folder_string,objects2save,pinn)