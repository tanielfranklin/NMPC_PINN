# NMPC_PINN

Resources <br>
Casadi-based simulator of an ESP oil lift system based in pavlov model <br>
NMPC routine for setpoint tracking<br>
NMPC routines for economic optimization (Nok)<br>
A Physics informed neural network model for ESP-lifted oil well system<br>

PINN VFM <br>
 If this system help you, you are encouraged to cite the following paper:<br>
    @article{franklin2022,
      title     = {A Physics-Informed Neural Networks (PINN) oriented approach to flow metering in oil wells: an ESP lifted oil well system as a case study},
      author    = { Franklin,Taniel S.  and  Souza, Leonardo S. and  Fontes, Raony M. and  Martins, Márcio A. F.},
      year      = {2022},
      volume={5},
      journal = {Digital Chemical Engineering}
    } 

## Citation
 If this system help you, you are encouraged to cite the following paper:<br>

    @article{franklin2022,
      title     = {A Physics-Informed Neural Networks (PINN) oriented approach to flow metering in oil wells: an ESP lifted oil well system as a case study},
      author    = { Franklin,Taniel S.  and  Souza, Leonardo S. and  Fontes, Raony M. and  Martins, Márcio A. F.},
      year      = {2022},
      volume={5},
      journal = {Digital Chemical Engineering}
    } 


There are some datasets available. We build dataset01 (using main_build_dataset.py) for training purposes and dataset_opera (using main_build_dataset.py) to verify the model generability. <br>

The procedure to achieve a good model is: <br>
a) start the training using model_adam_200 (main_train_adam.py)  <br>
b) complete the training changing to main_train_lbfgs which results in model_adam_lbfgs model <br>
c) pay attention to the loss terms evolution to define a better set of weights <br>

The notebook pinn_operation_prediction.ipynb demonstrates the model capability during free prediction <br>

Routines to train with Adam and L-BFGS optimizers (requires tensor flow probability)<br>
Routines to restore models trained previously <br>

Feel free to contribute with software improvements.

