

class VarHistory(object):
    def __init__(self):
        self.steps = []
        self.rho_train = []
        self.PI_train = []
        self.best_rho = 0
        self.best_PI = 0
        self.best_step = 0
        self.Font=14
    def append(self, step, rho_train, PI_train):
        self.steps.append(step)
        self.rho_train.append(rho_train)
        self.PI_train.append(PI_train)
    def update_best(self,trainstate):    
        self.best_step = trainstate.best_step
        self.best_rho = trainstate.rho.numpy()
        self.best_PI = trainstate.PI.numpy()

    