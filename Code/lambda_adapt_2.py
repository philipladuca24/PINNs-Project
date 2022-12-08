import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import optimizers
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from jaxopt import ScipyMinimize
from jax.scipy.optimize import minimize
from functools import partial

#pip install jaxopt
import jaxopt

# CLEAN UP CODE AND ADD DOCSTRINGS 

def random_layer_params(m, n, key, scale):
    w_key, b_key = random.split(key)
    # might want to initialize with glorot?
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    # why is this scaling used?
    return [
        random_layer_params(m, n, k, 2.0 / (jnp.sqrt(m + n)))
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]

@jit
def predict(params, X):
    activations = X
    for w, b in params[:-1]:
        activations = tanh(jnp.dot(activations, w) + b) 
    final_w, final_b = params[-1]
    logits = jnp.sum(jnp.dot(activations, final_w) + final_b)
    return logits

@jit
def net_u(params, X):

    x_array = jnp.array([X])
    return predict(params, x_array)


def net_ux(params):
  
    def ux(X):
        return grad(net_u, argnums=1)(params, X)
    
    return jit(ux)

def net_uxx(params):

    def uxx(X):
        u_x = net_ux(params)
        return grad(u_x)(X) 
    
    return jit(uxx)

@jit
def funx(X):
    return jnp.exp(X)

@jit
def loss_f(params, X, nu):
  u = vmap(net_u, (None, 0))(params, X) 
  u_xxf = net_uxx(params)
  u_xx = vmap(u_xxf, (0))(X)
  fx = vmap(funx, (0))(X)
  res = nu * u_xx - u - fx
  loss_f = jnp.mean((res.flatten()) ** 2)
  return loss_f

@jit 
def loss_lb(params):
    loss_lb = (net_u(params, -1)-1) ** 2 
    return loss_lb

@jit 
def loss_ub(params):
    loss_ub = (net_u(params, 1)) ** 2
    return loss_ub

@jit
def loss(params, X, nu, l_lb, l_ub):
    lossf = loss_f(params, X, nu)
    losslb = loss_lb(params)
    lossub = loss_ub(params)
    return jnp.sum(l_lb * losslb + l_ub * lossub + lossf)

@jit
def step_param(istep, opt_state, X, lb, ub):
    params = get_params(opt_state)
    g = grad(loss, argnums=0)(params, X, nu, lb, ub)
    return opt_update(istep, g, opt_state)

@jit
def step_lb(istep, params, X, opt_state, ub):
    lb = get_params_lb(opt_state)
    g = grad(loss, argnums=3)(params, X, nu, lb, ub)
    return opt_update_lb(istep, -g, opt_state)

@jit
def step_ub(istep, params, X, lb, opt_state):
    ub = get_params_ub(opt_state)
    g = grad(loss, argnums=4)(params, X, nu, lb, ub)
    return opt_update_ub(istep, -g, opt_state)

def param_flatten(params):
    params_new = jnp.array([])
    for m in range(4):
        for y in range(2):
            a = jnp.ravel(params[m][y])
            params_new = jnp.concatenate([params_new,a])
    params_new = jnp.array(params_new)
    return params_new

def param_reshape(params, sizes):
    params_re = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        a = params[:m*n]
        a = jnp.reshape(a, (m, n))
        b = params[m*n:m*n + n]
        params = params[m*n + n :]
        params_re.append((jnp.array(a),jnp.array(b)))
    return params_re

def loss_scipy(params, X, nu, l_lb, l_ub, sizes):
    params_re = param_reshape(params, sizes)
    return loss(params_re, X, nu, l_lb, l_ub)

def minimize_lbfgs(params, X, nu, lb, ub, sizes): 
    minimizer = jaxopt.LBFGS(fun=loss_scipy, jit=False)
    opt_params = minimizer.run(params, X, nu, lb, ub, sizes)
    opt_params = opt_params.params
    return opt_params

####### Hyperparameters ##################
nu = 10 ** (-3)
layer_sizes = [1, 20, 20, 20, 1]
nIter = 20000 + 1
params = init_network_params(layer_sizes, random.PRNGKey(0))
lambda_lb = random.uniform(random.PRNGKey(0), shape=[1])
lambda_ub = random.uniform(random.PRNGKey(0), shape=[1])
opt_init, opt_update, get_params = optimizers.adam(5e-4)
opt_state = opt_init(params)
opt_init_lb, opt_update_lb, get_params_lb = optimizers.adam(5e-4)
opt_state_lb = opt_init_lb(lambda_lb)
opt_init_ub, opt_update_ub, get_params_ub = optimizers.adam(5e-4)
opt_state_ub = opt_init_ub(lambda_ub)
l_lb_list = []
l_ub_list = []
lf_list = []
x = jnp.arange(-1, 1.1, 0.1)

pbar = trange(nIter)

lam_lb_list = []
lam_ub_list = []

for it in pbar:
    opt_state = step_param(it, opt_state, x, lambda_lb, lambda_ub)
    opt_state_lb = step_lb(it, params, x, opt_state_lb, lambda_ub)
    opt_state_ub = step_ub(it, params, x, lambda_lb, opt_state_ub)

    if it % 1 == 0:
        params = get_params(opt_state)
        lambda_lb = get_params_lb(opt_state_lb)
        lambda_ub = get_params_ub(opt_state_ub)
        l_lb = loss_lb(params)
        l_ub = loss_ub(params)
        l_f = loss_f(params, x, nu)

        pbar.set_postfix({"loss_res": l_f, "loss_bound": l_lb + l_ub})

        l_lb_list.append(l_lb)
        l_ub_list.append(l_ub)
        lf_list.append(l_f)
        lam_lb_list.append(lambda_lb)
        lam_ub_list.append(lambda_ub)

params_new = param_flatten(params)
params_min = minimize_lbfgs(params_new, x, nu, lambda_lb, lambda_ub, layer_sizes)
#params_min = minimize(loss_scipy, params_new, args=(x, nu, lambda_lb, lambda_ub, layer_sizes), method="BFGS")

print(lambda_lb, lambda_ub)
params_mini = param_reshape(params_min, layer_sizes)
#params_mini = param_reshape(params_min.x, layer_sizes)
u_pred = vmap(predict, (None, 0))(params_mini, x)

plt.plot(x, u_pred)

# plt.plot(lam_lb_list)
# plt.plot(lam_ub_list)
# plt.plot(l_lb_list)
# plt.plot(l_ub_list)
# plt.plot(lf_list)

plt.show()

print(loss_lb(params), "loss lower")
print(loss_ub(params), "loss upper")

print(loss_lb(params_mini), "loss lower_l-bfgs")
print(loss_ub(params_mini), "loss upper_l-bfgs")


# np.save("ld_list.npy", np.array(lb_list), allow_pickle=True)
# np.save("lf_list.npy", np.array(lf_list), allow_pickle=True)