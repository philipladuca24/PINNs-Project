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
import sys
sys.path.insert(0, 'Utilities/')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from matplotlib.pyplot import savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax
import time


"""
    Notes // Todo list:

    24/11/22

        - Need to find way to load and save models for Jax, current models take ~12min to train. 
        Potential solution: https://github.com/google/flax/discussions/1876 Would need to be able to 
        toggle between retraining and loading at will. Considering we aren't using flax, we might
        want to save /load models via pickle instead: 
        https://stackoverflow.com/questions/64550792/how-do-i-save-an-optimizer-state-of-jax-trained-model
        Jax doesn't have native support for loading and saving models, so we will need to either use
        other libraries or we could save the weights and biases in a list format and then write a script 
        in which we initialize the model with the saved weights and biases.  
        Fix: See run_saved_model.py for more information.
        Note: Will need to separate the model builder from the visualiser in this doc.


        - TypeError: error message below. Error probably coming from plt.figure(1.0, 1.1) on line 236.
        Likely stemming from latent error of "ImportError: cannot import name 'newfig' from 'plotting'". 
        Tried to solve using: https://github.com/maziarraissi/PINNs/issues/36
        Ultimately 'solved' by commented out the import line (line 17) and replaced it with line 18. 
        Newfig doesn't appear in the matplotlib documentation so thought it was referring to plt.figure 
        instead. Note: plotting seems to be an import from matplotlib.pyplot but not extremely clear either.
        "
        Traceback (most recent call last):
            File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/pinn_bugers_jax.py", line 201, in <module>
            File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/_api/deprecation.py", line 454, in wrapper
                return func(*args, **kwargs)
            File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/pyplot.py", line 783, in figure
                manager = new_figure_manager(
            File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/pyplot.py", line 359, in new_figure_manager
                return _get_backend_mod().new_figure_manager(*args, **kwargs)
            File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/backend_bases.py", line 3504, in new_figure_manager
                fig = fig_cls(*args, **kwargs)
            File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/_api/deprecation.py", line 454, in wrapper
                return func(*args, **kwargs)
            File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/figure.py", line 2473, in __init__
                self.bbox_inches = Bbox.from_bounds(0, 0, *figsize)
        TypeError: Value after * must be an iterable, not float
        "
        
        - TypeError: error message below. Error is a continuation of the error above.
        "
            Traceback (most recent call last):
                File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/pinn_bugers_jax.py", line 241, in <module>
            TypeError: cannot unpack non-iterable Figure object
        "

"""


def random_layer_params(m, n, key, scale):
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key, (m, n)), jnp.zeros(n)

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, 2.0/(jnp.sqrt(m+n))) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


@jit
def predict(params, X, lb, ub):
  # per-example predictions
  H =  2.0*(X - lb)/(ub - lb) - 1.0
  for w, b in params[:-1]:
    H = tanh(jnp.dot(H, w) + b)
  final_w, final_b = params[-1]
  H = jnp.dot(H, final_w) + final_b
  return H

@jit
def net_u(params, x, t, lb, ub):
    x_con =jnp.array([x, t])
    y_pred = predict(params, x_con, lb, ub)
    return y_pred


@jit
def net_u_grad(params, x, t, lb, ub):
    x_con =jnp.array([x, t])
    y_pred = predict(params, x_con, lb, ub)
    #print(f"shape y_pred: {jnp.shape(y_pred)}")
    return y_pred[0]

@jit
def loss_data(params,x,t, lb, ub, u_train):
    u_pred = vmap(net_u, (None, 0, 0, None, None))(params, x, t, lb, ub)
    print(f"Shape of u_pred: {jnp.shape(u_pred)}")
    #sys.exit()
    loss = jnp.mean((u_pred - u_train)**2 )
    return loss

def net_f(params, lb, ub):
    def u_t(x, t):
        ut = grad(net_u_grad, argnums=2)(params, x, t, lb, ub) 
        return ut

    def u_x(x, t):
        ux = grad(net_u_grad, argnums=1)(params, x, t, lb, ub) 
        return ux   
    return jit(u_t), jit(u_x)

def net_fxx(params, lb, ub):
    def u_xx(x, t):
        _, u_x = net_f(params, lb, ub) 
        ux = grad(u_x, argnums=0)(x, t) 
        return ux   
    return jit(u_xx)


@jit
def loss_f(params, x, t, lb, ub, nu):
    u = vmap(net_u, (None, 0, 0, None, None))(params, x, t, lb, ub)
    u_tf, u_xf = net_f(params, lb, ub)
    u_xxf = net_fxx(params, lb, ub)
    u_t = vmap(u_tf, (0, 0))(x, t)
    u_x = vmap(u_xf, (0, 0))(x, t)
    u_xx = vmap(u_xxf, (0, 0))(x, t)
    res = u_t + u.flatten() * u_x - nu * u_xx 
    loss_f = jnp.mean((res.flatten())**2)
    return loss_f

@jit
def predict_u(params, x_star, t_star, lb, ub):
    u_pred = vmap(net_u, (None, 0, 0, None, None))(params, x_star, t_star, lb, ub)
    return u_pred



if __name__ == "__main__":

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
    
    
    params = init_network_params(layers, random.PRNGKey(1234))

   
    
    opt_init, opt_update, get_params = optimizers.adam(5e-4)
    opt_state = opt_init(params)
    nIter = 20000 + 1
    ld_list = []
    lf_list = []

    def loss_fn(params, x_f, t_f,x_d, t_d, lb, ub, nu, y_d):
        loss_res = 0.01*loss_f(params, x_f, t_f, lb, ub, nu)
        data_loss = loss_data(params, x_d, t_d, lb, ub, y_d) 
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
            l_d = loss_data(params, x_d, t_d, lb, ub, u_train)
            l_f = loss_f(params, x_f, t_f, lb, ub, nu)
            pbar.set_postfix({'Loss': l_d, 'loss_physics': l_f})
            ld_list.append(l_d)
            lf_list.append(l_f)

    l_list = []

    u_pred = predict_u(params, x_star, t_star, lb, ub)
    print(f"u_pred Shape: {u_pred.shape}")
            
    error_u = jnp.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))
    np.save("ld_list.npy", np.array(ld_list), allow_pickle=True) 
    np.save("lf_list.npy", np.array(lf_list), allow_pickle=True)  
 
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)


    
    
    fig, ax = plt.figure(1)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)

    savefig("Burgers")
    






   