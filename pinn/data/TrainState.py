import numpy as np
class TrainState(object):
    def __init__(self):
        self.epoch, self.step = 0, 0

        self.sess = None
        self.best_weights = None

        # Data
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        # Results of current step
        self.y_pred_train = None
        self.loss_train,self.loss_train_bc, self.loss_train_f, self.loss_test = None, None,None,None
        self.loss_train_x1,self.loss_train_x2,self.loss_train_x3=None, None,None
        self.y_pred_test, self.y_std_test = None, None
        self.metrics_test = None
        self.weights = None
        self.rho = None
        self.PI = None


        # The best results corresponding to the min train loss
        self.best_step = None
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_y, self.best_ystd = None, None
        self.best_metrics = None
        self.best_rho = self.rho
        self.best_PI = self.PI
        # The best results corresponding to the min test loss
        self.best_test_train=np.inf
        self.best_test_loss=np.inf
        self.best_test_step = self.step
        self.best_test_train = np.sum(self.loss_train)
        self.best_test_weights=self.weights
        self.best_test_rho = self.rho
        self.best_test_PI = self.PI


    def set_tfsession(self, sess):
        self.sess = sess

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train, self.y_train, self.train_aux_vars = (
            X_train,
            y_train,
            train_aux_vars,
        )

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test, self.y_test, self.test_aux_vars = X_test, y_test, test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_weights=self.weights
            self.best_rho = self.rho
            self.best_PI = self.PI
            self.best_loss_test = self.loss_test
        if self.best_test_loss > self.loss_test:
            self.best_test_step = self.step
            self.best_test_train = np.sum(self.loss_train)
            self.best_test_weights=self.weights
            self.best_test_rho = self.rho
            self.best_test_PI = self.PI
            self.best_test_loss = self.loss_test
            # self.best_y, self.best_ystd = self.y_pred_test, self.y_std_test
            # self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf

    def packed_data(self):
        def merge_values(values):
            if values is None:
                return None
            return np.hstack(values) if isinstance(values, (list, tuple)) else values

        X_train = merge_values(self.X_train)
        y_train = merge_values(self.y_train)
        X_test = merge_values(self.X_test)
        y_test = merge_values(self.y_test)
        best_y = merge_values(self.best_y)
        best_ystd = merge_values(self.best_ystd)
        return X_train, y_train, X_test, y_test, best_y, best_ystd