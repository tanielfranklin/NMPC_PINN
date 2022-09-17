from matplotlib.legend import Legend
Font=14
class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.loss_train_bc=[]
        self.loss_train_f=[]
        self.loss_train_x1=[]
        self.loss_train_x2=[]
        self.loss_train_x3=[]
        self.metrics_test = []
        self.loss_weights = 1
        self.Font=14

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train,loss_train_bc,loss_train_f,loss_train_x1,loss_train_x2,loss_train_x3, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        self.loss_train_bc.append(loss_train_bc)
        self.loss_train_f.append(loss_train_f)
        self.loss_train_x1.append(loss_train_x1)
        self.loss_train_x2.append(loss_train_x2)
        self.loss_train_x3.append(loss_train_x3)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        # if metrics_test is None:
        #     metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
    



