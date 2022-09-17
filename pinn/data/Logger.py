import tensorflow as tf
import time
import numpy as np
import datetime
class Logger(object):
  def __init__(self, frequency=10):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))

    self.start_time =int(time.time())
    self.frequency = frequency

  def __get_elapsed(self):
     elap=(int(time.time()) - self.start_time)
     return f"{int(elap/60):2d}:{int(elap%60):2d}"

  def __get_error_u(self):
    return self.error_fn()

  def set_error_fn(self, error_fn):
    self.error_fn = error_fn
  
  def log_train_start(self, model):
    print("\nTraining started")
    print("================")
    self.model = model
    print(f"PINN Mode:{self.model.pinn_mode}")
    print(f"Learning rate=:{np.array(model.optimizer.learning_rate):.4f}")
    print(self.model.summary_model())


  def log_train_epoch(self, epoch, loss,loss_f=0.0,loss_bc=0.0, custom="", is_iter=False):
    if epoch % self.frequency == 0:
        #print(f"{epoch:6d} {self.__get_elapsed()} {loss:.4e} {loss_f:.4e} ")
        # print '000050 00:06 4.1441e-03 4.1441e-03 0.0000e+00 3.6360e-01'
        #print("==================")
        print(f"{epoch:6d} {self.__get_elapsed()} {loss:.3e} {loss_bc:.3e} {loss_f:.3e} {self.__get_error_u():.3e}"+custom)
        #print(f"{'epoch' if is_iter else 'epoch'} = {epoch:6d}  elap={self.__get_elapsed()}  loss = {loss:.4e} loss_bc={loss_bc:.4e} loss_EDO={loss_f:.4e}  error = {self.__get_error_u():.4e}  " + custom)
  def log_train_opt(self, name,mode,w):
    # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
    print(f"—— Starting {name} optimization —— Pinn_mode:{mode}")
    print(f'==============Weights===============')
    print(f'[ wbc ,  w1 ,  w2 ,  w3 ]')
    print(f'{w}')
    print('                                                    |==== Weigthed Residues ===|ODE parameters|')
    print('==============================================================================================================================')
    print('epoch | elap|  Total  | Loss BC |Loss ODE |  Test   | w1r1   | w2r2   |  w3r3  | rho |   PI   |')
    print('==============================================================================================================================')

#--------------------------------------------------------------------------------------##############################2#####

  def log_train_end(self, epoch, train_state, custom=""):
    print("==================")
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()}  Test = {self.__get_error_u():.4e}  " + custom)
    #print(f"Best model at epoch: {train_state.best_step} best loss: {train_state.best_loss_train: .3e} best loss test {train_state.best_loss_test:.3e}"  )