import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian
from jax import random
from jax.example_libraries.optimizers import adam
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.optimize import minimize
from jaxlib.xla_extension import DeviceArray
from functools import partial

"""
Dynamically adapts the weighting (lambda) of the different components of the loss function.
ie train Self-Adaptation Weights for the initial, boundary, and residue points, as well as 
the network weights.

'A key feature of self-adaptive PINNs is that the loss L(w, λr, λb, λ0) is minimized with 
respect to the network weights w, as usual, but is maximized with respect to the self-adaptation
weights λr , λb , λ0 , i.e., the objective is: min max L(w, λr, λb, λ0).' 
- L. D. McClenny, U. Braga-Neto 2022.

In this 1D implementation, the self-adaptive weights are defined for boundary and residue only.

Args:
    x_lb (int): x lower bound.
    x_ub (int): x upper bound.
    x (DeviceArray): Collocation points in the domain.
    layers (list[int]): Network architecture.
    nu (float): _description_
"""


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
    # activations = X
    # for w, b in params[:-1]:
    #     activations = tanh(jnp.matmul(activations, w) + b)

    # final_w, final_b = params[-1]
    # return jnp.sum(jnp.matmul(activations, final_w) + final_b)

    activations = X
    params = list(params)
    print(len(params), "length of params")
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
    X = jnp.array([X])
    return predict(params, X)


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
        return jacobian(net_u, argnums=1)(params, X)

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
        return jacobian(u_x)(X)

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
    loss_b = lambda_b * ((net_u(params, lb) - 1) ** 2 + (net_u(params, ub)) ** 2)
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
    loss_f = lambda_f * jnp.mean((res.flatten()) ** 2)
    return loss_f


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
    return lossb + lossf


@partial(jit, static_argnums=(0,))
def optimise_net(istep: int, opt_state, X, lambda_b, lambda_f):
    """
    Computes gradients and applies the optimizer (Adam) to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): Collocation points in the domain.
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        (Tracer of) DeviceArray: Optimised network parameters.
    """
    params = get_params(opt_state)
    g = jacobian(loss)(params, X, nu, lambda_b, lambda_f)
    opt_state = opt_update(istep, g, opt_state)
    return opt_state


@jit
def optimise_lambda(params, X, nu, lambda_b, lambda_f):
    """
    Computes gradients and applies the optimizer (L-BFGS-B) to the self-adaptive weights (lambda).

    Args:
        params (Tracer of list[DeviceArray]): list containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): _description_
        lambda_b (Tracer of DeviceArray): Self-adaptive weight for the boundary loss.
        lambda_f (Tracer of DeviceArray): Self-adaptive weight for the residue loss.

    Returns:
        Tuple[(Tracer of) DeviceArray]: Tuple of the optimised self-adaptive weights.
    """
    lamb = minimize(fun=loss_b, x0=lambda_b, args=(lambda_b, ), method="BFGS")
    lamf = minimize(fun=loss_f, x0=lambda_f, args=(lambda_f,), method="BFGS")
    return lamb, lamf


####### Hyperparameters ##################
nu = 10 ** (-3)
layer_sizes = [1, 20, 20, 20, 1]
nIter = 20000 + 1

lb = -1
ub = 1

params = init_network_params(layer_sizes, random.PRNGKey(0))
opt_init, opt_update, get_params = adam(0.001)
lb_list = []
lf_list = []

x = jnp.arange(-1, 1.1, 0.1)
# x = jnp.expand_dims(x, axis=1)

# defining lambdas
NF = x.shape[0]
lambda_f = jnp.squeeze(jnp.array(jnp.reshape(np.repeat(100, NF), (NF, -1))), 1)
lambda_b = jnp.squeeze(random.uniform(random.PRNGKey(0), shape=[NF, 1]), 1)


# define optimizer, layers, hyperparameters, and loss lists


"""
Training the model.

Returns:
    Tuple[list[DeviceArray]]: Predicted u(x) and lists containing the boundary and
    residue loss respectively.
"""

pbar = trange(nIter)
opt_state = opt_init(params)

for it in pbar:
    opt_state = optimise_net(it, opt_state, x, lambda_b, lambda_f)
    if it % 1 == 0:
        opt_params_net = get_params(opt_state)
        opt_state_lambda = optimise_lambda(opt_params_net, x, nu, lambda_b, lambda_f)

        lambda_b = opt_state_lambda[0]
        lambda_f = opt_state_lambda[1]

        l_b = loss_b(opt_params_net, lambda_b)
        l_f = loss_f(opt_params_net, x, nu, lambda_f)
        pbar.set_postfix({"Loss_res": l_f, "loss_bound": l_b})
        lb_list += l_b
        lf_list += l_f

u_pred = vmap(predict, (None, 0))(opt_params_net, x)

plt.plot(x, u_pred)
plt.plot(nIter, lb_list)
plt.plot(nIter, lf_list)
plt.show()
