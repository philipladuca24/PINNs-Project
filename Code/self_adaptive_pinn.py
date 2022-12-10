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


"""
Dynamically adapts the weighting (lambda) of the different components of the loss function.
ie train Self-Adaptation Weights for the initial, boundary, and residue points, as well as 
the network weights.

'A key feature of self-adaptive PINNs is that the loss L(w, λr, λb, λ0) is minimized with 
respect to the network weights w, as usual, but is maximized with respect to the self-adaptation
weights λr , λb , λ0 , i.e., the objective is: min max L(w, λr, λb, λ0).' 
- L. D. McClenny, U. Braga-Neto 2022.

In this 1D implementation, the self-adaptive weights are only defined for the upper and lower bound loss.

Args: # do 
    x_lb (int): x lower bound.
    x_ub (int): x upper bound.
    x (DeviceArray): Collocation points in the domain.
    layers (list[int]): Network architecture.
    nu (float): _description_
"""
# do we need to modify the args above? 


# ----------------------------------------------------------------------------------------------------------------
# FEEDFORWARD NEURAL NETWORK ARCHITECTURE 
# ----------------------------------------------------------------------------------------------------------------

def random_layer_params(m, n, key, scale):
    """
    A helper function to randomly initialize weights and biases for a
    dense neural network layer.

    Args:
        m (int): Shape of our weights (MxN matrix).
        n (int): Shape of our weights (MxN matrix).
        key (DeviceArray): Key for random modules.
        scale (DeviceArray[float]): Float value between 0 and 1 to scale the initialisation.

    Returns:
        DeviceArray: Randomised initialisation for a single layer.
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)


def init_network_params(sizes, key):
    """
    Initialize all layers for a fully-connected neural network with
    sizes "sizes".

    Args:
        sizes (list[int]): Network architecture.
        key (DeviceArray): Key for random modules.

    Returns:
        list[DeviceArray]: Fully initialised network parameters.
    """
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, 2.0 / (jnp.sqrt(m + n)))
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


@jit
def predict(params, X):
    """
    Per example predictions.

    Args:
        params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        X (Tracer of DeviceArray): Single point in the domain.

    Returns:
        Tracer of DeviceArray: Output predictions (u_pred)
    """
    activations = X
    for w, b in params[:-1]:
        activations = tanh(jnp.dot(activations, w) + b) 
    final_w, final_b = params[-1]
    logits = jnp.sum(jnp.dot(activations, final_w) + final_b)
    return logits


@jit
def net_u(params, X):
    """
    Define neural network for u(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        Tracer of DeviceArray: u(x).
    """
    x_array = jnp.array([X])
    return predict(params, x_array)


def net_ux(params):
    """
    Define neural network for first spatial derivative of u(x): u'(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        Tracer of DeviceArray: u'(x).
    """
    def ux(X):
        return grad(net_u, argnums=1)(params, X)
    
    return jit(ux)


def net_uxx(params):
    """
    Define neural network for second spatial derivative of u(x): u''(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        Tracer of DeviceArray: u''(x).
    """
    def uxx(X):
        u_x = net_ux(params)
        return grad(u_x)(X) 
    return jit(uxx)


@jit
def funx(X):
    """
    The f(x) in the partial derivative equation.

    Args:
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        Tracer of DeviceArray: Elementwise exponent of X
    """
    return jnp.exp(X)

@jit
def loss_f(params, X, nu):
    """
    Calculates our reside loss.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): _description_

    Returns:
        Tracer of DeviceArray: Residue loss.
    """
    u = vmap(net_u, (None, 0))(params, X) 
    u_xxf = net_uxx(params)
    u_xx = vmap(u_xxf, (0))(X)
    fx = vmap(funx, (0))(X)
    res = nu * u_xx - u - fx
    loss_f = jnp.mean((res.flatten()) ** 2)
    return loss_f


@jit 
def loss_lb(params):
    """
    Calculates the lower bound loss.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.

    Returns:
        Tracer of DeviceArray: Lower bound loss.
    """
    loss_lb = (net_u(params, -1)-1) ** 2 
    return loss_lb


@jit 
def loss_ub(params):
    """
    Calculates the upper bound loss.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.

    Returns:
        Tracer of DeviceArray: Lower bound loss.
    """
    loss_ub = (net_u(params, 1)) ** 2
    return loss_ub


@jit
def loss(params, X, nu, l_lb, l_ub):
    """
    Combines the lower bound, upper bound, and residue loss into a single loss matrix.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): _description_
        l_lb (Tracer of DeviceArray): Self-adaptive weight for the lower bound loss.
        l_ub (Tracer of DeviceArray): Self-adaptive weight for the upper bound loss.

    Returns:
        Tracer of DeviceArray: Total loss matrix.
    """
    lossf = loss_f(params, X, nu)
    losslb = loss_lb(params)
    lossub = loss_ub(params)
    return jnp.sum(l_lb * losslb + l_ub * lossub + lossf)


@jit
def step_param(istep, opt_state, X, lb, ub):
    """
    Training step that computes gradients for network weights and applies the optimizer 
    (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): Collocation points in the domain.
        lb (Tracer of DeviceArray): Self-adaptive weight for the lower bound loss.
        ub (Tracer of DeviceArray): Self-adaptive weight for the upper bound loss.

    Returns:
        (Tracer of) DeviceArray: Optimised network parameters.
    """
    params = get_params(opt_state)
    g = grad(loss, argnums=0)(params, X, nu, lb, ub)
    return opt_update(istep, g, opt_state)


@jit
def step_lb(istep, params, X, opt_state, ub):
    """
    Training step that computes gradients for self-adaptive weight for lower bound and
    applies the optimizer (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        opt_state (Tracer of OptimizerState): Optimised self-adaptive weight for lower bound loss. 
        ub (Tracer of DeviceArray): Self-adaptive weight for the upper bound loss.

    Returns:
        (Tracer of) DeviceArray: Optimised self-adaptive weight for lower bound.
    """
    lb = get_params_lb(opt_state)
    g = grad(loss, argnums=3)(params, X, nu, lb, ub)
    return opt_update_lb(istep, -g, opt_state)


@jit
def step_ub(istep, params, X, lb, opt_state):
    """
    Training step that computes gradients for self-adaptive weight for upper bound and
    applies the optimizer (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        lb (Tracer of DeviceArray): Self-adaptive weight for the lower bound loss.
        opt_state (Tracer of OptimizerState): Optimised self-adaptive weight for upper bound loss. 

    Returns:
        (Tracer of) DeviceArray: Optimised self-adaptive weight for upper bound.
    """
    ub = get_params_ub(opt_state)
    g = grad(loss, argnums=4)(params, X, nu, lb, ub)
    return opt_update_ub(istep, -g, opt_state)


def param_flatten(params):
    # PHILIP ADD DOCSTRINGS HERE 
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

# ----------------------------------------------------------------------------------------------------------------
# MODEL HYPERPARAMETERS  
# ----------------------------------------------------------------------------------------------------------------

nu = 10 ** (-3) # multiplicative constant 
layer_sizes = [1, 20, 20, 20, 1] # input, output, hidden layer sizes 
nIter = 20000 + 1 # number of epochs/iterations 

params = init_network_params(layer_sizes, random.PRNGKey(0)) # initialising weights and biases 
lambda_lb = random.uniform(random.PRNGKey(0), shape=[1]) # initialising lower bound self-adaptive weight 
lambda_ub = random.uniform(random.PRNGKey(0), shape=[1]) # initialising upper bound self-adaptive weight 

# optimizers for weights/biases, upper bound adaptive weight, and lower bound adaptive weight 
opt_init, opt_update, get_params = optimizers.adam(5e-4)
opt_state = opt_init(params)
opt_init_lb, opt_update_lb, get_params_lb = optimizers.adam(5e-4)
opt_state_lb = opt_init_lb(lambda_lb)
opt_init_ub, opt_update_ub, get_params_ub = optimizers.adam(5e-4)
opt_state_ub = opt_init_ub(lambda_ub)

# lists to keep track over upper bound, lower bound, and residue loss during training 
l_lb_list = []
l_ub_list = []
lf_list = []
# lists to keep track of upper and lower self-adaptive weight values during training 
lam_lb_list = []
lam_ub_list = []

# generation of input data or collocation points 
x = jnp.arange(-1, 1.05, 0.05)


# ----------------------------------------------------------------------------------------------------------------
# MODEL TRAINING  
# ----------------------------------------------------------------------------------------------------------------

pbar = trange(nIter)
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

# ----------------------------------------------------------------------------------------------------------------
# FINAL PARAMETER OPTIMIZATION 
# ----------------------------------------------------------------------------------------------------------------

params_new = param_flatten(params)
params_min = minimize_lbfgs(params_new, x, nu, lambda_lb, lambda_ub, layer_sizes)
params_mini = param_reshape(params_min, layer_sizes)

# final prediction of u(x) 
u_pred = vmap(predict, (None, 0))(params_mini, x)

# ----------------------------------------------------------------------------------------------------------------
# PLOTTING 
# ----------------------------------------------------------------------------------------------------------------

fig, axs = plt.subplots(1, 3)

axs[0].plot(x, u_pred, label="'$\lambda_1$': 1.873, '$\lambda_2$': 2.326")
axs[0].set_title('Self-Adaptive PINN Proposed Solution')
axs[0].set_xlabel("x")
axs[0].set_ylabel("Predicted u(x)")
axs[0].legend()

axs[1].plot(l_lb_list, label='Lower Bound loss')
axs[1].plot(l_ub_list, label="Upper Bound loss")
axs[1].plot(lf_list, label='Residue loss')
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()
axs[1].set_title('Residue and Function Loss vs. Epochs')

axs[2].plot(lam_lb_list, label="'$\lambda_1$': Lower bound penalty", color="blue")
axs[2].plot(lam_ub_list, label="'$\lambda_2$': Upper bound penalty", color="purple")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Lambda value")
axs[2].legend()
axs[2].set_title('Optimised Lambda Values vs. Epochs')

plt.show()

# ----------------------------------------------------------------------------------------------------------------
# FINAL LAMBDAS, LOSSES, AND OPTIMIZED LOSSES   
# ----------------------------------------------------------------------------------------------------------------

print(lambda_lb, lambda_ub)
print(loss_lb(params), "loss lower")
print(loss_ub(params), "loss upper")
print(loss_lb(params_mini), "loss lower_l-bfgs")
print(loss_ub(params_mini), "loss upper_l-bfgs")