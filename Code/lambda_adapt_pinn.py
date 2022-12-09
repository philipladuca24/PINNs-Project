import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian
from jax import random
from jax.example_libraries.optimizers import adam
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.optimize import minimize
from functools import partial

"""
Dynamically adapts the weighting (lambda) of the different components of the loss function.
ie train Self-Adaptation Weights for the initial, boundary, and residue points, as well as 
the network weights.

'A key feature of self-adaptive PINNs is that the loss L(w, λr, λb, λ0) is minimized with 
respect to the network weights w, as usual, but is maximized with respect to the self-adaptation
weights λr , λb , λ0 , i.e., the objective is: min max L(w, λr, λb, λ0).' 
- L. D. McClenny, U. Braga-Neto 2022.

In this 1D implementation, the self-adaptive weights are only defined for the upper and lower bound loss.

Args:
    x_lb (int): x lower bound.
    x_ub (int): x upper bound.
    x (DeviceArray): Collocation points in the domain.
    layers (list[int]): Network architecture.
    nu (float): _description_
"""

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

    # activations = X
    # params = list(params)
    # for w, b in params[:-1]:
    #     activations = tanh(jnp.dot(activations, w) + b)
    # final_w, final_b = params[-1]
    # logits = jnp.sum(jnp.dot(activations, final_w) + final_b)
    # return logits


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
        return (grad(net_u, argnums=1)(params, X))[0]

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
def loss_b(params, lambda_b):
    """
    Calculates our boundary loss.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.

    Returns:
        Tracer of DeviceArray: Boundary loss.
    """
    loss_b = (net_u(params, lb) - 1) ** 2 + (net_u(params, ub)) ** 2
    return loss_b


@jit
def loss_f(params, X, nu, lambda_f):
    """
    Calculates our reside loss.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): _description_
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        Tracer of DeviceArray: Residue loss.
    """
    u = vmap(net_u, (None, 0))(params, X)
    u_xxf = net_uxx(params)
    u_xx = vmap(u_xxf, (0))(X)
    fx = vmap(funx, (0))(X)
    res = nu * u_xx - u - fx
    loss_f = jnp.mean((res.flatten())**2)
    return loss_f # how to do the loss properly


@jit
def loss(params, X, nu, lambda_b, lambda_f):
    """
    Combines the boundary and residue loss into a single loss matrix.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): _description_
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        Tracer of DeviceArray: Total loss matrix.
    """
    lossf = loss_f(params, X, nu, lambda_f)
    lossb = loss_b(params, lambda_b)
    return 0.01*lossb + lossf

@jit
def optimise_net(istep: int, opt_state, X, lambda_b, lambda_f):
    """
    Computes gradients for network weights and applies the optimizer (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): Collocation points in the domain.
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        (Tracer of) DeviceArray: Optimised network parameters.
    """
    params = get_params_net(opt_state)
    #allow_int=True ?
    g = grad(loss, argnums=0)(params, X, nu, lambda_b, lambda_f)
    return opt_update_net(istep, g, opt_state)

@jit
def optimise_lamb(istep: int, params, X, opt_state, lambda_f):
    """
    Computes gradients for lower bound self-adaptive weight and applies the optimizer (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): Collocation points in the domain.
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        (Tracer of) DeviceArray: Optimised network parameters.
    """
    lambda_b = get_params_lamb(opt_state)
    g = grad(loss, argnums=3)(params, X, nu, lambda_b, lambda_f)
    return opt_update_lamb(istep, -g, opt_state)

@jit
def optimise_lamf(istep: int, params, X, lambda_b, opt_state):
    """
    Computes gradients for self-adaptive weights and applies the optimizer (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): Collocation points in the domain.
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        (Tracer of) DeviceArray: Optimised network parameters.
    """
    lambda_f = get_params_lamf(opt_state)
    g = grad(loss, argnums=4)(params, X, nu, lambda_b, lambda_f)
    return opt_update_lamf(istep, -g, opt_state)

@jit
def optimise_lbfgs(params, X, nu, lambda_b, lambda_f):
    """
    Computes gradients and applies the optimizer (L-BFGS-B) to the loss.
    The adaptive weights are only updated in the Adam training steps, 
    and are held constant during L-BFGS training.

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): _description_
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        Tuple[(Tracer of) DeviceArray]: Tuple of the optimised loss.
    """
    opt_params = minimize(fun=loss, x0=jnp.array([params, X, nu, lambda_b, lambda_f]), args=(lambda_b, lambda_f), method="BFGS")
    opt_params = opt_params.x
    # lamf = minimize(fun=loss_f, x0=lambda_f, args=(X, nu, lambda_f,), method="BFGS")
    # lamf = lamf.x
    return opt_params


# ----------------------------------------------------------------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------------------------------------------------------------

####### Hyperparameters ##################
nu = 10 ** (-3)
layer_sizes = [1, 20, 20, 20, 1]
nIter = 20000 + 1

lb = -1
ub = 1

params = init_network_params(layer_sizes, random.PRNGKey(0))
opt_init_net, opt_update_net, get_params_net = adam(0.001)
opt_init_lamb, opt_update_lamb, get_params_lamb = adam(0.001)
opt_init_lamf, opt_update_lamf, get_params_lamf = adam(0.001)
lb_list = []
lf_list = []

x = jnp.arange(-1, 1.1, 0.1)
x = jnp.expand_dims(x, axis=1)

# defining lambdas
NF = x.shape[0]
lambda_f = jnp.squeeze(random.uniform(random.PRNGKey(0), shape=[NF, 1]), 1)
lambda_b = jnp.squeeze(random.uniform(random.PRNGKey(0), shape=[NF, 1]), 1)


"""
Training the model.

Returns:
    Tuple[list[DeviceArray]]: Predicted u(x) and lists containing the boundary and
    residue loss respectively.
"""

pbar = trange(nIter)
opt_state_net = opt_init_net(params)
opt_state_lamb = opt_init_lamb(lambda_b)
opt_state_lamf = opt_init_lamf(lambda_f)

for it in pbar:
    opt_state_net = optimise_net(it, opt_state_net, x, lambda_b, lambda_f)
    opt_state_lamb = optimise_lamb(it, params, x, opt_state_lamb, lambda_f)
    opt_state_lamf = optimise_lamf(it, params, x, lambda_b, opt_state_lamf)

    if it % 1 == 0:
        params = get_params_net(opt_state_net)
        lambda_b = get_params_lamb(opt_state_lamb)
        lambda_f = get_params_lamf(opt_state_lamf)
        l_b = loss_b(params, lambda_b)
        l_f = loss_f(params, x, nu, lambda_f)
        pbar.set_postfix({"Loss_res": l_f, "loss_bound": l_b})
        lb_list.append(l_b)
        lf_list.append(l_f)

init_guess = [lb_list[-1]+ lf_list[-1], None]
print(init_guess)
# fin_params = optimise_lbfgs(opt_params_net, x, nu, opt_state_lamb, opt_state_lamf)
u_pred = vmap(predict, (None, 0))(params, x)

plt.plot(x, u_pred)
plt.show()
