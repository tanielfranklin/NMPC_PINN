import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from data.utils import restore_pinn_model, plot_test
from data.utils import plot_states_BCS, plot_u
from data.BCS_casadi import BCS_model
from data.parameters import Parameters
# from data.TrainingReport import TrainingReport
from data.PinnPredictor import PinnPredictor
with open("dataset_opera.pk", 'rb') as open_file:
    ds = pickle.load(open_file)
predictor=PinnPredictor("pinn_models/data_model02")
BCS_parameters=Parameters()
# steady-state conditions
xss = np.vstack([8311024.82175957, 2990109.06207437,
                0.00995042241351780, 50., 50.])
nx = 5
nu = 2
ny = 2

uss = np.vstack([50., 50.])
yss = np.vstack([6000142.88550200, 592.126490003812])
# output scale factor

Ts = 2  # sampling time
# --------------------------------------------------------------------------
# constraints on inputs
# --------------------------------------------------------------------------
umin = np.vstack([35, 0])    # lower bounds of inputs
umax = np.vstack([65, 100])    # upper bounds of inputs
# maximum variation of input moves: [0.5 Hz/s; 0.5 #/s]
dumax = Ts*np.vstack([0.5, 0.5])
# ----------------------------------------------
# ------Normalização através dos pesos----------
q = np.hstack([100, 1]) / (yss**2)  # weights on controlled variables
q = np.hstack([1e6, 1e8]) / (yss**2)  # weights on controlled variables
r = np.array([100, 1]) / (uss.T**2)  # weights on control actions
qu = 1000 / (uss[1]**2)

print("Instancia BCS")
bcs_init = [nu, nx, ny, Ts, umin, umax, dumax]
bcs = BCS_model(nu, nx, ny, Ts, umin, umax, dumax)

# Uncomment to see dataset figures
# ds.gen_fig()
# plt.show()

# exit()

#Initial conditions for PINN predictions
xi=xss[0:2].T #bar
pr=126e5
pm=20e5
ui=np.vstack([uss,pm,pr]).T



Ui,Xi=predictor.start_dataset(ui,xi)
y,yout=predictor.predict_pinn(Ui,Xi)
print(yout)
ui[:,0]=60.
Ui,Xi=predictor.update_inputs(y,ui,Ui,Xi)
y,yout=predictor.predict_pinn(Ui,Xi)
print(yout)

exit()
y,yout=predictor.predict_pinn(Ui,Xi)
print(yout)
exit()

nsim=100
y0,_=predictor.model(Ui,Xi)
predict=y0
nsteps=ds.n_steps_in
for i in range(nsim):
        Xi=tf.concat([Xi[:,1:,:],y0[:,:,:-1]],1) # Remove older time-step and update states vector with new predictions (remove q)
        uk=ds.un[nsteps+i:nsteps+i+1,:]
        Ui=tf.concat([Ui[:,1:,:],np.array([uk])],1) # Remove older time-step and update exogenous vector with the next time-step
        y0=predictor.model(tf.concat([Ui,Xi],2)) # Compute the next prediction
        predict=tf.concat([predict,y0],0) # Store        
predict=predict.numpy()
ti=0;to=ti+nsim+1

y=ds.train_y_full[ti:to,:,:]
u=ds.train_X[ti:to,0,:4]
t=np.arange(0,nsim+1,1)
plot_test(y, predict,[BCS_parameters.xc,BCS_parameters.x0])
plt.show()
