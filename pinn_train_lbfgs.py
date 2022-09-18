

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from data.pinn_BCS import pinn_vfm

from data.Logger import Logger
from data.utils import Struct, save_model_files, restore_pinn_model
from data.TrainingReport import TrainingReport

# time = np.linspace(0, maxtime, 200) # Regular points inside the domain

with open("dataset01.pk", 'rb') as open_file:
    ds = pickle.load(open_file)

# ========================================
# # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_config = Struct()
# Positive integer. The number of iterations allowed to run in parallel.
nt_config.parallel_iter = 2
# The maximum number of iterations for L-BFGS updates.
nt_config.maxIter = 400
# Specifies the maximum number of (position_delta, gradient_delta) correction pairs to keep as implicit approximation of the Hessian matrix.
nt_config.nCorrection = 50
# If the relative change in the objective value between one iteration and the next is smaller than this value, the algorithm is stopped.
nt_config.tolfun = 1e-5
# Specifies the gradient tolerance for the procedure. If the supremum norm of the gradient vector is below this number, the algorithm is stopped.
# Specifies the gradient tolerance for the procedure. If the supremum norm of the gradient vector is below this number, the algorithm is stopped.
nt_config.tol = 1e-5

# ---------------------------------------

# ========================================
# Creating the model and training
logger = Logger(frequency=100)
# logger.set_error_fn(error)
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
neurons = 15
rho = 950
PI = 2.32*1e-9
start_rho = 0.9*950/rho
start_PI = 0.9*2.32*1e-9/PI
var = [start_rho, start_PI]  # parâmetros normalizados
n_features = 6  # Network inputs  (fk, zc,pmc,pr, x1,x2)
nt_config.maxIter = 100
Nc = 10
pinn = pinn_vfm(Nc, tf_optimizer, logger,
                var=var, pinn_mode="on",
                inputs=n_features,
                n_steps_in=ds.n_steps_in,
                n_steps_out=ds.n_steps_out,
                parameters=ds.parameters)
local = "pinn_models/model_adam_200/"
pinn.u_model.load_weights(local+'model.h5')
pinn_restored = restore_pinn_model(local)
# #######################################
# pinn.lamb_l1 = tf.constant(1.0, dtype=tf.float32)  # x1 residue weight
# pinn.lamb_l2 = tf.constant(1.0, dtype=tf.float32)  # x3 residue weight
# pinn.lamb_l3 = tf.constant(1.0, dtype=tf.float32)  # x3 residue weight
# nt_config.maxIter = 200
# loss_history, trainstate, var_history=pinn.fit_LBFGS(ds.lbfgs_dataset(), nt_config)
# training_report = TrainingReport(pinn, [loss_history, trainstate, var_history], ds)
# ------------------------ Set new weights --------------------------------
pinn.lamb_l1 = tf.constant(2.0, dtype=tf.float32)  # x3 residue weight
pinn.lamb_l2 = tf.constant(1.9, dtype=tf.float32)  # x3 residue weight
pinn.lamb_l3 = tf.constant(0.01, dtype=tf.float32)  # x3 residue weight
# ------------------------ Remaining epochs with LBFGS --------------------------------
loss_history, trainstate, var_history = pinn.fit_LBFGS(
    ds.lbfgs_dataset(), nt_config)
training_report = TrainingReport(
    pinn, [loss_history, trainstate, var_history], ds)
training_report.gen_plot_result()
training_report.gen_var_plot()
training_report.gen_plot_loss_res()
plt.show()
# ------------------------ Saving files --------------------------------
# Uncomment the lines below to save the model
folder_string = "pinn_models/model_adam_lbfgs"
objects2save = {"Loss": loss_history,
                "trainstate": trainstate, "vartrain": var_history}
save_model_files(folder_string, objects2save, pinn)
