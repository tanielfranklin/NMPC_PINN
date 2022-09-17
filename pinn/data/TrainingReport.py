
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

Font=14
class TrainingReport():
    def __init__(self,model,training_data,ds):
        self.loss,trainstate,self.var_history=training_data
        self.pack_plot=ds.pack_plot
        self.x0=ds.parameters.x0
        self.xc=ds.parameters.xc
        self.obs=ds.pack[0][:,0,:]
        self.pred_train, self.pred_test=self.__prep_data_plot(model.u_model) 
        
    
    
    def gen_plot_loss_res(self):
        # Plot the loss terms evolution
        loss_train = self.loss.loss_train
        loss_train_bc = self.loss.loss_train_bc
        loss_train_f = self.loss.loss_train_f
        loss_train_x1 = self.loss.loss_train_x1
        loss_train_x2 = self.loss.loss_train_x2
        loss_train_x3 = self.loss.loss_train_x3
        loss_test = self.loss.loss_test

        Fig=plt.figure(figsize=(10, 4))
        ax1=Fig.add_subplot(1,1,1)
        ax1.semilogy(self.loss.steps, loss_train,'--k', label="Training loss")
        ax1.semilogy(self.loss.steps, loss_train_x1,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_1}$")
        ax1.semilogy(self.loss.steps, loss_train_bc, label="$\mathcal{L}_{\mathbf{BC}}$")
        ax1.semilogy(self.loss.steps, loss_train_x2,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_2}$")
        #plt.semilogy(self.loss.steps, loss_train_f,'--k', label="ode")
        ax1.semilogy(self.loss.steps, loss_test,'-k',lw=2, label="Validation loss")
        ax1.semilogy(self.loss.steps, loss_train_x3,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_3}$")
        ax1.grid()

        # for i in range(len(losshistory.metrics_test[0])):
        #     plt.semilogy(
        #         losshistory.steps,
        #         np.array(losshistory.metrics_test)[:, i],
        #         label="Test metric",
        #     )
        plt.setp(ax1.get_xticklabels(), fontsize=Font)
        plt.setp(ax1.get_yticklabels(), fontsize=Font)
        plt.xlabel("Epochs",fontsize=Font)
        plt.legend(bbox_to_anchor=(0.4, 1.0), ncol = 3,fontsize=Font)
        return Fig
        
    def __prep_data_plot(self,model):
        train_X,_, test_X , _=self.pack_plot
        y_pred_train=model.predict(train_X)
        y_pred_train=y_pred_train.reshape(y_pred_train.shape[0]*y_pred_train.shape[1],y_pred_train.shape[2])
        y_pred_test=model.predict(test_X)
        y_pred_test=y_pred_test.reshape(y_pred_test.shape[0]*y_pred_test.shape[1],y_pred_test.shape[2])
        n_steps_out=1
        for i in range(y_pred_train.shape[0]):
            k=i*n_steps_out
            if k==0:
                pred_train=y_pred_train[k:k+1,:]
            pred_train=np.vstack((pred_train,y_pred_train[k:k+1,:]))
            if k==y_pred_train.shape[0]:
                #print(k)
                break
        
        for i in range(y_pred_test.shape[0]):
            k=i*n_steps_out
            if k==0:
                pred_test=y_pred_test[k:k+1,:]
            pred_test=np.vstack((pred_test,y_pred_test[k:k+1,:]))
            if k==y_pred_train.shape[0]:
                break
        return pred_train*self.xc+self.x0, pred_test*self.xc+self.x0
    def gen_plot_result(self):
        #plot predictions with training data and test data   
        if self.obs.shape[-1]==2:
            self.obs=self.obs*self.xc[0:2]+self.x0[0:2]
        if self.obs.shape[-1]==3:
            self.obs=self.obs*self.xc+self.x0
            #y_mean = np.mean(prediction1)
    
        k = np.arange(0,len(self.obs))
        ktr=np.arange(0,len(self.pred_train))
        kts=np.arange(len(self.pred_train),len(self.pred_train)+len(self.pred_test))
        #print(k.shape,ktr.shape,kts.shape)
        Fig=plt.figure(figsize=(10, 4))
        label=["$P_{bh}(bar)$","$P_{wh}(bar)$","$q (m^3/h)$"]
        scale=[1/1e5,1/1e5,3600]
        cor=["black","black","gray"]
        leg_lb=['Observed data','Observed data','No measured data']


        for i,val in enumerate(label):
            ax1=Fig.add_subplot(len(label),1,i+1)
            if i==2:
                l0, =ax1.plot(k, self.obs[:,i]*scale[i],"-",color=cor[i])
            else:
                l1, =ax1.plot(k, self.obs[:,i]*scale[i],"-",color=cor[i])

            l2, =ax1.plot(ktr, self.pred_train[:,i]*scale[i],":",color='red',lw=2)
            l3, =ax1.plot(kts, self.pred_test[:,i]*scale[i],":",color='blue',lw=2)
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
    
    def gen_var_plot(self,lim_rho=None,lim_PI=None):
        rho_train = self.var_history.rho_train
        PI_train = self.var_history.PI_train
        steps=self.var_history.steps
        rho_ref=950
        conv=3.6e8;
        PI_ref=2.32e-9*conv
        PI_train = [element * conv for element in PI_train]
        label=[r"Estimated $\rho$", 'Estimated PI', r"True $\rho$", "True PI"]

        #fig=plt.figure()
        fig=plt.figure(figsize=(7, 4))
        ax=fig.add_subplot()
        ln1=ax.plot(steps,rho_train,'-k', label=label[0])
        ln3=ax.plot(steps,np.ones((len(steps),1))*rho_ref,'--k',label=label[2])
        #ln5=plt.annotate("X", (int(self.best_step),self.best_rho))
        #ln5=ax.plot(varhistory.best_step,varhistory.best_rho,'k',marker='x',label=r'Best $\rho$')
        
        #ax.set(ylim=(800, 1200))
        #ax2.set(ylim=(0.72, 0.9 ))
        plt.setp(ax.get_xticklabels(), fontsize=Font)
        plt.setp(ax.get_yticklabels(), fontsize=Font)
        ax2=ax.twinx()

        ax2.set_ylabel(r'$PI~[m^3/h/bar]$',fontsize=Font)
        ln2=ax2.plot(steps,PI_train,'-',color='gray', label=label[1])
        ln4=ax2.plot(steps,np.ones_like(steps)*PI_ref,'--',color='gray', label=label[3])
        #plt.annotate("X", (int(self.best_step),self.best_PI*conv))
        #ln6=ax2.plot(varhistory.best_step,varhistory.best_PI*conv,'k',marker='x' ,label='Best PI')
        
        

        # if PI_lim!=None:
        #ax2.set(ylim=(0.72, 0.9 ))
        if lim_rho!=None:
            ax.set(ylim=(lim_rho[0], lim_rho[1]))
        if lim_PI!=None:    
            ax2.set(ylim=(lim_PI[0], lim_PI[1] ))
        # added these three lines
        # ln = ln1+ln2#+ln2+ln3
        
        # labs = [l.get_label() for l in ln]
        #ax2.legend(ln, labs, loc='lower right')#,ncol=2)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles2, labels2, loc='lower right',frameon=False)#,ncol=2)
        plt.setp(ax2.get_xticklabels(), fontsize=Font)
        plt.setp(ax2.get_yticklabels(), fontsize=Font)
        leg = Legend(ax2, handles, labels,loc='lower center', frameon=False)
        ax2.add_artist(leg)
        ax.set_ylabel(r"$\rho ~[kg/m^3]$",fontsize=Font)
        ax.set_xlabel('Epochs',fontsize=Font)
        plt.grid(True)
        #plt.show()
        return fig
    
    