import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import numpy as np
import scipy.io
from pyDOE import lhs
from jax.nn import relu, tanh, relu
import sys
from jax import jacfwd, jacrev
from jax.example_libraries import optimizers
from tqdm import trange
from functools import partial
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class BurgersModel(object):
    def __init__(self) -> None:
        pass

    def random_layer_params(self, m, n, key, scale):
        w_key, b_key = random.split(key)
        return scale*random.normal(w_key, (m, n)), jnp.zeros(n)

    def init_network_params(self, sizes, key):
        keys = random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k, 2.0/(jnp.sqrt(m+n))) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


    @jit
    def predict(self, params, X, lb, ub):
        # per-example predictions
        H =  2.0*(X - lb)/(ub - lb) - 1.0
        for w, b in params[:-1]:
            H = tanh(jnp.dot(H, w) + b)
        final_w, final_b = params[-1]
        H = jnp.dot(H, final_w) + final_b
        return H

    @jit
    def net_u(self, params, x, t, lb, ub):
        x_con =jnp.array([x, t])
        y_pred = self.predict(params, x_con, lb, ub)
        return y_pred


    @jit
    def net_u_grad(self, params, x, t, lb, ub):
        x_con =jnp.array([x, t])
        y_pred = self.predict(params, x_con, lb, ub)
        #print(f"shape y_pred: {jnp.shape(y_pred)}")
        return y_pred[0]

    @jit
    def loss_data(self, params,x,t, lb, ub, u_train):
        u_pred = vmap(self.net_u, (None, 0, 0, None, None))(params, x, t, lb, ub)
        print(f"Shape of u_pred: {jnp.shape(u_pred)}")
        #sys.exit()
        loss = jnp.mean((u_pred - u_train)**2 )
        return loss

    def net_f(self, params, lb, ub):
        def u_t(x, t):
            ut = grad(self.net_u_grad, argnums=2)(params, x, t, lb, ub) 
            return ut

        def u_x(self, x, t):
            ux = grad(self.net_u_grad, argnums=1)(params, x, t, lb, ub) 
            return ux   
        return jit(u_t), jit(u_x)

    def net_fxx(self, params, lb, ub):
        def u_xx(x, t):
            _, u_x = self.net_f(params, lb, ub) 
            ux = grad(u_x, argnums=0)(x, t) 
            return ux   
        return jit(u_xx)


    @jit
    def loss_f(self, params, x, t, lb, ub, nu):
        u = vmap(self.net_u, (None, 0, 0, None, None))(params, x, t, lb, ub)
        u_tf, u_xf = self.net_f(params, lb, ub)
        u_xxf = self.net_fxx(params, lb, ub)
        u_t = vmap(u_tf, (0, 0))(x, t)
        u_x = vmap(u_xf, (0, 0))(x, t)
        u_xx = vmap(u_xxf, (0, 0))(x, t)
        res = u_t + u.flatten() * u_x - nu * u_xx 
        loss_f = jnp.mean((res.flatten())**2)
        return loss_f

    @jit
    def predict_u(self, params, x_star, t_star, lb, ub):
        u_pred = vmap(self.net_u, (None, 0, 0, None, None))(params, x_star, t_star, lb, ub)
        return u_pred



    def call(self, params, x_star, t_star, lb, ub):

        nu = 0.01/np.pi
        N_u = 100
        N_f = 10000
        layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        data = scipy.io.loadmat('../Data/burgers_shock.mat')
        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        Exact = np.real(data['usol']).T    
        X, T = np.meshgrid(x,t)
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_star = Exact.flatten()[:,None]              

        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)    
        xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
        uu1 = Exact[0:1,:].T
        xx2 = np.hstack((X[:,0:1], T[:,0:1]))
        uu2 = Exact[:,0:1]
        xx3 = np.hstack((X[:,-1:], T[:,-1:]))
        uu3 = Exact[:,-1:]
        X_u_train = np.vstack([xx1, xx2, xx3])
        X_f_train = lb + (ub-lb)*lhs(2, N_f)
        #print(X_u_train)

        X_f_train = np.vstack((X_f_train, X_u_train))
        u_train = np.vstack([uu1, uu2, uu3])
        idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx,:]

        x_d = X_u_train[:, 0]
        t_d = X_u_train[:, 1]
        
        x_f = X_f_train[:, 0]
        t_f = X_f_train[:, 1]

        x_star = X_star[:, 0]
        t_star = X_star[:, 1]


        params = self.init_network_params(layers, random.PRNGKey(1234))

        opt_init, opt_update, get_params = optimizers.adam(5e-4)
        opt_state = opt_init(params)
        nIter = 20000 + 1
        ld_list = []
        lf_list = []

        def loss_fn(params, x_f, t_f,x_d, t_d, lb, ub, nu, y_d):
            loss_res = 0.01*self.loss_f(params, x_f, t_f, lb, ub, nu)
            data_loss = self.loss_data(params, x_d, t_d, lb, ub, y_d) 
            return loss_res + data_loss

        #@partial(jit, static_argnums=(0,))
        @jit
        def step(istep, opt_state, t_d, x_d, y_d, t_f, x_f, lb, ub):
            param = get_params(opt_state) 
            g = grad(loss_fn, argnums=0)(param, x_f, t_f,x_d, t_d, lb, ub, nu, y_d)
            return opt_update(istep, g, opt_state)

        pbar = trange(nIter)
        
        for it in pbar:
            opt_state = step(it, opt_state, t_d, x_d, u_train, t_f, x_f, lb, ub)
            if it % 1 == 0:
                params = get_params(opt_state)
                l_d = self.loss_data(params, x_d, t_d, lb, ub, u_train)
                l_f = self.loss_f(params, x_f, t_f, lb, ub, nu)
                pbar.set_postfix({'Loss': l_d, 'loss_physics': l_f})
                ld_list.append(l_d)
                lf_list.append(l_f)

        l_list = []

        u_pred = self.predict_u(params, x_star, t_star, lb, ub)
        print(f"u_pred Shape: {u_pred.shape}")
                
        error_u = jnp.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
        print('Error u: %e' % (error_u))
        np.save("ld_list.npy", np.array(ld_list), allow_pickle=True) 
        np.save("lf_list.npy", np.array(lf_list), allow_pickle=True)  
    
        
        U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        Error = np.abs(Exact - U_pred)
        return U_pred, Error