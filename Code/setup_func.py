import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import optimizers
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def random_layer_params(m, n, key, scale):
    """
    A helper function to randomly initialize weights and biases for a 
    dense neural network layer.

    Args:
        m (_type_): _description_
        n (_type_): _description_
        key (_type_): _description_
        scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    w_key, b_key = random.split(key)
    # might want to initialize with glorot?
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)

def init_network_params(sizes, key):
    """
    Initialize all layers for a fully-connected neural network with 
    sizes "sizes".

    Args:
        sizes (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """
    keys = random.split(key, len(sizes))
    # why is this scaling used?
    return [
        random_layer_params(m, n, k, 2.0 / (jnp.sqrt(m + n)))
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]

@jit
def predict(params, X):
    """
    Per example predictions

    Args:
        params (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
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
def loss_b(params):
    loss_b = (net_u(params, -1) - 1) ** 2 + (net_u(params, 1)) ** 2
    return loss_b

@jit
def loss(params, X, nu):
    lossf = loss_f(params, X, nu)
    lossb = loss_b(params)
    return 0.01*lossf + lossb

####### Hyperparameters ##################
nu = 10 ** (-3)
layer_sizes = [1, 20, 20, 20, 1]
nIter = 20000 + 1
params = init_network_params(layer_sizes, random.PRNGKey(0))
opt_init, opt_update, get_params = optimizers.adam(5e-4)
opt_state = opt_init(params)
lb_list = []
lf_list = []
x = jnp.arange(-1, 1, 0.01)

# we can try to increase the layer size or increase the number of points being trained on, we can also try to 
# include 1 in the arange, different weighting on the loss function may help, we can try glorot
# initialization for the parameters, not sure how to be more careful with the boundry


@jit
def step(istep, opt_state, X):
    param = get_params(opt_state)
    g = grad(loss, argnums=0)(param, X, nu)
    return opt_update(istep, g, opt_state)


pbar = trange(nIter)

for it in pbar:
    opt_state = step(it, opt_state, x)
    if it % 1 == 0:
        params = get_params(opt_state)
        l_b = loss_b(params)
        l_f = loss_f(params, x, nu)
        pbar.set_postfix({"Loss_res": l_f, "loss_bound": l_b})
        lb_list.append(l_b)
        lf_list.append(l_f)

x_pred = vmap(net_u, (None, 0))(params, x)

plt.plot(x, x_pred)
# plt.plot(nIter, lb_list)
# plt.plot(nIter, lf_list)
plt.show()


# np.save("ld_list.npy", np.array(lb_list), allow_pickle=True)
# np.save("lf_list.npy", np.array(lf_list), allow_pickle=True)


