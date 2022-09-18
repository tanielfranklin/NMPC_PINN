import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

def plot_test(y,yp,norm):
    #ind defines timestep of output to plot
    xc,x0=norm
    xc,x0=xc[:3],x0[:3]
    #yp=model.predict(entradas)
    
    MSE= [np.mean(np.square(y[:,0,0]- yp[:,0,0])),
        np.mean(np.square(y[:,0,1]- yp[:,0,1])),
        np.mean(np.square(y[:,0,2]- yp[:,0,2]))]
    MSE_all=np.mean(np.square(y[:,0,:]- yp[:,0,:]))
    yp=yp[:,0,:]
    y=y[:,0,:]
    yp=np.vstack([yp[i,:]*xc.T+x0.T for i in range(yp.shape[0])])
    #yp=(yp*xc.T+x0.T)
    y=np.vstack([y[i,:]*xc.T+x0.T for i in range(y.shape[0])])
    k=np.arange(y.shape[0])
    #print(y[:,0:1].shape)
    plt.figure(figsize=(20, 4))
    Fig=plt.figure()
    Fig.suptitle(f"Test Data MSE = [{MSE_all:.1e}] , [{MSE[0]:.1e}, {MSE[1]:.1e}, {MSE[2]:.1e}]")
    #plt.title(f"Test Data , MSE = [{MSE[0]:.1e}, {MSE[1]:.1e}, {MSE[2]:.1e}]") 
    #plt.title("Test Data from {} to {} , Mean = {:.2f}".format(start, end, y_mean) ,  fontsize=18)
    ax1=Fig.add_subplot(3,1,1)
    #ax1.plot(y[:,0:1],"k:",linewidth=2)
    ax1.plot(y[:,0:1]/1e5,"k:",linewidth=2)
    #ax1.plot(yp[:,0]/1e5, label='pred')
    ax1.plot(yp[:,0:1]/1e5)
    ax1.set_ylabel("Pbh",  fontsize=Font)
    ax1.set_xticklabels([])
    plt.setp(ax1.get_yticklabels(), fontsize=Font)
    ax1.grid(True)
    plt.grid(True)
    ax2=Fig.add_subplot(3,1,2)
    ax2.set_ylabel("Pwh",  fontsize=Font)
    ax2.plot(y[:,1]/1e5,"k:",linewidth=2)
    #ax2.plot(yp[:,1]/1e5, label='pred')
    ax2.plot(yp[:,1]/1e5)
    ax2.set_xticklabels([])
    plt.setp(ax2.get_yticklabels(), fontsize=Font)
    ax2.grid(True)
    #plt.grid(True)
    ax3=Fig.add_subplot(3,1,3)
    ax3.set_ylabel("q",  fontsize=Font)
    ax3.plot(y[:,2:]*3600,"k:",linewidth=2,label='obs')
    #ax3.plot(yp[:,2:]*3600, label='pred')
    ax3.plot(yp[:,2:]*3600, label='pred')
    #ax3.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    #ax3.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.setp(ax3.get_xticklabels(), fontsize=Font)
    plt.setp(ax3.get_yticklabels(), fontsize=Font)
         
    ax3.set_xlabel('Time(s)' ,  fontsize=Font)
    plt.legend(bbox_to_anchor=(1.0, -0.3), ncol = 3)
    # label_texts= [label.get_text() for label in ax3.xaxis.get_ticklabels()]
    # print(label_texts)
    return Fig

def save_model_files(folder_string,dict_of_objects,model):
    os.makedirs(folder_string, exist_ok=True)    
    for keys,item in dict_of_objects.items():
        store_model_files([folder_string+"/"+keys+".pk"],[item])

    model_json = model.u_model.to_json()
    with open(folder_string+"/"+"model.json", "w") as json_file:
        json_file.write(model_json)
    model.u_model.save_weights(folder_string+"/"+"model.h5")
    print(f"Saved in {folder_string}")





def test_res(y,u,model):
    model.rho=tf.Variable(1.0, dtype=tf.float32)
    model.PI=tf.Variable(1.0, dtype=tf.float32)
    r1,r2,r3=model.ED_BCS(y,u)
    return r1.numpy(),r2.numpy(),r3.numpy()

def dydt(y_pred,ts):
    #Central 3 pontos
    y = y_pred[:,0,:]
    n=y.shape[0]
    try:
        if n<6:
            raise Exception("Model output size must have at least 6 time points ")          
    except Exception as inst:
        print(inst.args)
        raise
    #Progressiva e regressiva 3 pontos
    pro3=tf.constant([[-3,4,-1]],dtype=tf.float32)/(2*ts)
    reg3=tf.constant([[1,-4,3]],dtype=tf.float32)/(2*ts)
    d1=tf.matmul(pro3,y[0:3,:])
    #print(d1)
    dn=tf.matmul(reg3,y[-3:,:])
    #Central 2 pontos
    dc=(y[2:n,:]-y[0:n-2,:])/(2*ts)        
    return tf.concat([d1,dc,dn],axis=0)


def plot_states_BCS(input,t,norm=False): 
    scale=np.array([1/1e5,1/1e5,3600])
    if norm==True:
        scale=np.array([1,1,1])
    fig4=plt.figure()
    label = ['Pbh','Pbw','q'];
    for i,val in enumerate(label): 
        ax1=fig4.add_subplot(len(label),1,i+1)   
        ax1.plot(t ,input[:,i]*scale[i], label=val)
        if i!=2:
            ax1.set_xticklabels([])
        ax1.set_ylabel(val)
        plt.grid(True)
    return fig4
def plot_u(uplot):
    fig=plt.figure()
    label = ['f','z','pman','pr'];
    for i,val in enumerate(label):
        ax=fig.add_subplot(len(label),1,i+1)
        ax.plot(uplot[i])#, label=val)
        ax.set_ylabel(val)
        if i!=3:
            ax.set_xticklabels([])
        plt.grid(True)
    return fig

def add_noise_norm(signal,sigma):
    n = np.random.normal(0, sigma, len(signal))
    n=n.reshape([len(n),1])
    sig=(1+n)*signal
    return sig

def add_noise(signal,snr_db):
  # Set a target SNR
  target_snr_db = snr_db
  # Calculate signal power and convert to dB 
  sig_avg_watts = np.mean(signal**2)
  sig_avg_db = 10 * np.log10(sig_avg_watts)
  # Calculate noise according to [2] then convert to watts
  noise_avg_db = sig_avg_db - target_snr_db
  noise_avg_watts = 10 ** (noise_avg_db / 10)
  # Generate an sample of white noise
  mean_noise = 0
  noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
  # Noise up the original signal
  noise_volts=noise_volts.reshape([len(signal),1])
  return signal + noise_volts




# loc_drive='/content/drive/MyDrive/Dados_BCS/'

# import sys
# sys.path.append('/drive/MyDrive/Dados_BCS/')


# exec(compile(open(loc_drive+'subrotinas.py', "rb").read(), loc_drive+'subrotinas.py', 'exec'))
Font=14 # size of plot text
def store_model_files(file_name_list,obj_list): 
    for i in np.arange(len(file_name_list)):
        with open(file_name_list[i], "wb") as open_file:
            pickle.dump(obj_list[i], open_file)
   
#     print("Saved files")

def load_model_data(file_name):
    with open(file_name, 'rb') as open_file:
        obj = pickle.load(open_file)
    return obj

def restore_pinn_model(local):
    return [load_model_data(local+'Loss.pk'), load_model_data(local+'trainstate.pk'),load_model_data(local+'vartrain.pk')]

def plot_result(pred_train, pred_test, obs,xc,x0):   
    if obs.shape[-1]==2:
        obs=obs*xc[0:2]+x0[0:2]
    if obs.shape[-1]==3:
        obs=obs*xc+x0

    #y_mean = np.mean(prediction1)
    
    k = np.arange(0,len(obs))
    ktr=np.arange(0,len(pred_train))
    kts=np.arange(len(pred_train),len(pred_train)+len(pred_test))
    #print(k.shape,ktr.shape,kts.shape)
    Fig=plt.figure(figsize=(10, 4))
    label=["$P_{bh}(bar)$","$P_{wh}(bar)$","$q (m^3/h)$"]
    scale=[1/1e5,1/1e5,3600]
    cor=["black","black","gray"]
    leg_lb=['Observed data','Observed data','No measured data']


    for i,val in enumerate(label):
        ax1=Fig.add_subplot(len(label),1,i+1)
        if i==2:
            l0, =ax1.plot(k, obs[:,i]*scale[i],"-",color=cor[i])
        else:
            l1, =ax1.plot(k, obs[:,i]*scale[i],"-",color=cor[i])

        l2, =ax1.plot(ktr, pred_train[:,i]*scale[i],":",color='red',lw=2)
        l3, =ax1.plot(kts, pred_test[:,i]*scale[i],":",color='blue',lw=2)
        ax1.set_ylabel(val,  fontsize=Font)
        plt.setp(ax1.get_yticklabels(), fontsize=Font)
        if i!=2:
            ax1.set_xticklabels([])
        
        ax1.grid(True)
    plt.grid(True)
    ax1.set_xlabel('$Time(s)$' ,  fontsize=Font)
    plt.setp(ax1.get_xticklabels(), fontsize=Font)
    plt.setp(ax1.get_yticklabels(), fontsize=Font)
    plt.legend([l0,l1,l2,l3],['Non measured data','Observed data','Prediction with training data','Prediction with validation data'],bbox_to_anchor=(1.0, 4.2), ncol = 2,fontsize=Font)
    #plt.legend(bbox_to_anchor=(1, 3.8), ncol = 2)
    #fig.legend(handles=[l1, l2])
    return Fig



# Plot history and future
def plot_multistep(history, prediction1 , groundtruth , start , end):
    plt.figure(figsize=(20, 4))
    y_mean = np.mean(prediction1)
    range_history = len(history)
    range_future = list(range(range_history, range_history + len(prediction1[:,0])))
    Fig=plt.figure()
    #plt.title("Test Data from {} to {} , Mean = {:.2f}".format(start, end, y_mean) ,  fontsize=18)
    ax1=Fig.add_subplot(3,1,1)
    ax1.plot(np.arange(range_history), np.array(history[:,0]/1e5), label='History')
    ax1.plot(range_future, np.array(prediction1[:,0]/1e5),label='Forecasted with LSTM')
    ax1.plot(range_future, np.array(groundtruth[:,0]/1e5),":k",label='GroundTruth')
    ax1.set_ylabel("Pbh",  fontsize=Font)
    ax1.set_xticklabels([])
    ax1.grid(True)
    plt.grid(True)
    ax2=Fig.add_subplot(3,1,2)
    ax2.set_ylabel("Pwh",  fontsize=Font)
    ax2.plot(np.arange(range_history), np.array(history[:,1]/1e5), label='History')
    ax2.plot(range_future, np.array(prediction1[:,1]/1e5),label='Forecasted')
    ax2.plot(range_future, np.array(groundtruth[:,1]/1e5),":k",label='Observed')
    ax2.set_xticklabels([])
    ax2.grid(True)
    plt.grid(True)
    ax3=Fig.add_subplot(3,1,3)
    ax3.set_ylabel("q", fontsize=Font)
    ax3.plot(np.arange(range_history), np.array(history[:,2]*3600), label='History')
    ax3.plot(range_future, np.array(prediction1[:,2]*3600),label='Forecasted')
    ax3.plot(range_future, np.array(groundtruth[:,2]*3600),":k",label='Observed')
    plt.grid(True)
    #plt.legend(loc='upper left')    
    ax2.set_xlabel('Time(s)' ,  fontsize=Font)
    plt.legend(bbox_to_anchor=(0.9, -0.2), ncol = 3)









#@tf.function
def get_abs_max_grad(grad):
    r=np.zeros((len(grad))).astype(np.float32)
    for i in range(len(grad)):
        r[i]=tf.math.reduce_max(tf.abs(grad[i]))
    return tf.math.reduce_max(r)
#@tf.function
def get_abs_mean_grad(grad):
    r=np.zeros((len(grad))).astype(np.float32)
    for i in range(len(grad)):
        r[i]=tf.math.reduce_mean(tf.abs(grad[i]))
    return tf.math.reduce_mean(r)


# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)


def gen_traindata(file):
	data = np.load(file)
	return data["t"], data["x"], data["u"]
# time points

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value[0]] # concatena uma lista de	X com yhat (value)
	array = np.array(new_row) # converte para array
	array = array.reshape(1, len(array)) # Faz a transposta
	inverted = scaler.inverse_transform(array)	# reescala
	return inverted[0, -1] # retorna yhat (value) reescalonado




def forecast_on_batch(model, batch_size, X):
	X = X.reshape(batch_size, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
