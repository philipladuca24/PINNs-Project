import numpy as np
import scipy.io
from pyDOE import lhs

class BurgersPreprocessing():
    """
    Preprocessing for BurgersModel. Model construction etc should be done in runner.py
    Equation params set in here.
    Returns:
        X: Matrix representation of 'x' vectors
        T: Matrix representation of 't' vectors
        Exact: Real part of the 'u' solutions transposed

        X_star: hstack of flattened X and T matrices (each with a new axis of length one)
        u_star: flattened Exact (with a new axis of length one)
        lower_bound: lower bound of X_star
        upper_bound: upper bound of X_star
        u_train: Random rows and all columns of vstack of uu1,uu2,uu3 (from Exact)
        x_d: 0th column slice of X_u_train 
        t_d: 1st column slice of X_u_train
        x_f: 0th column slice of X_f_train (which combines X_f_train and X_u_train)
        t_f: 1st column slice of X_f_train 
        x_star: 0th column slice of X_star
        t_star: 1st column slice of X_star

    """
    def __init__(self):
        self.N_u = 100
        self.N_f = 10000
        data = scipy.io.loadmat('../Data/burgers_shock.mat')
        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        self.Exact = np.real(data['usol']).T    
        self.X, self.T = np.meshgrid(x,t)
        self.X_star = np.hstack((self.X.flatten()[:,None], self.T.flatten()[:,None]))
        self.u_star = self.Exact.flatten()[:,None]              
    def call(self):
        # Domain bounds
        lower_bound = self.X_star.min(0)
        upper_bound = self.X_star.max(0)    
        xx1 = np.hstack((self.X[0:1,:].T, self.T[0:1,:].T))
        uu1 = self.Exact[0:1,:].T
        xx2 = np.hstack((self.X[:,0:1], self.T[:,0:1]))
        uu2 = self.Exact[:,0:1]
        xx3 = np.hstack((self.X[:,-1:], self.T[:,-1:]))
        uu3 = self.Exact[:,-1:]
        X_u_train = np.vstack([xx1, xx2, xx3])
        X_f_train = lower_bound + (upper_bound-lower_bound)*lhs(2, self.N_f)
        #print(X_u_train)

        X_f_train = np.vstack((X_f_train, X_u_train))
        u_train = np.vstack([uu1, uu2, uu3])
        idx = np.random.choice(X_u_train.shape[0], self.N_u, replace=False) # acting like a dropout(?)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx,:]

        x_d = X_u_train[:, 0]
        t_d = X_u_train[:, 1]
        
        x_f = X_f_train[:, 0]
        t_f = X_f_train[:, 1]

        x_star = self.X_star[:, 0]
        t_star = self.X_star[:, 1]

        return self.X, self.T, self.Exact, self.X_star, self.u_star, lower_bound, upper_bound, u_train, x_d, t_d, x_f, t_f, x_star, t_star