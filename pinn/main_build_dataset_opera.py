import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data.BuildDataset import BuildingDataset
import pickle
from data.utils import store_model_files

# time = np.linspace(0, maxtime, 200) # Regular points inside the domain
# Treinamento e validação
dados_opera = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz')
dados=dados_opera
# Somente Validação

# building dataset for training
n_steps_in, n_steps_out = 20, 1  # convert into input/output many-to-one
split_point=int(0.98*dados["x1"].shape[0]) #percentage of data dedicated for training
ds=BuildingDataset(n_steps_in, n_steps_out, dados,split_point, batch_size=500)
ds.gen_fig()
file_name="dataset_opera.pk"
def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
save_object(ds,file_name)
ds.gen_fig()
plt.show()