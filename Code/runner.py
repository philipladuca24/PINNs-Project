import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import numpy as np
from pyDOE import lhs
from jax.nn import relu, tanh, relu
import sys
from jax.example_libraries import optimizers
from tqdm import trange
from functools import partial
from scipy.interpolate import griddata

from burgers_preprocessing import BurgersPreprocessing

"""
Script to run model initialised with loaded weights and biases.
"""


def random_layer_params(m, n, key, scale):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, 2.0 / (jnp.sqrt(m + n)))
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


@jit
def predict(params, X, lower_bound, upper_bound):
    # per-example predictions
    H = 2.0 * (X - lower_bound) / (upper_bound - lower_bound) - 1.0
    for w, b in params[:-1]:
        H = tanh(jnp.dot(H, w) + b)
    final_w, final_b = params[-1]
    H = jnp.dot(H, final_w) + final_b
    return H


@jit
def net_u(params, x, t, lower_bound, upper_bound):
    x_con = jnp.array([x, t])
    y_pred = predict(params, x_con, lower_bound, upper_bound)
    return y_pred


@jit
def net_u_grad(params, x, t, lower_bound, upper_bound):
    x_con = jnp.array([x, t])
    y_pred = predict(params, x_con, lower_bound, upper_bound)
    print(f"shape y_pred: {jnp.shape(y_pred)}")
    return y_pred[0]


@jit
def loss_data(params, x, t, lower_bound, upper_bound, u_train):
    u_pred = vmap(net_u, (None, 0, 0, None, None))(
        params, x, t, lower_bound, upper_bound
    )
    print(f"Shape of u_pred: {jnp.shape(u_pred)}")
    # sys.exit()
    loss = jnp.mean((u_pred - u_train) ** 2)
    return loss


def net_f(params, lower_bound, upper_bound):
    def u_t(x, t):
        ut = grad(net_u_grad, argnums=2)(params, x, t, lower_bound, upper_bound)
        return ut

    def u_x(x, t):
        ux = grad(net_u_grad, argnums=1)(params, x, t, lower_bound, upper_bound)
        return ux

    return jit(u_t), jit(u_x)


def net_fxx(params, lower_bound, upper_bound):
    def u_xx(x, t):
        _, u_x = net_f(params, lower_bound, upper_bound)
        ux = grad(u_x, argnums=0)(x, t)
        return ux

    return jit(u_xx)


@jit
def loss_f(params, x, t, lower_bound, upper_bound, nu):
    u = vmap(net_u, (None, 0, 0, None, None))(params, x, t, lower_bound, upper_bound)
    u_tf, u_xf = net_f(params, lower_bound, upper_bound)
    u_xxf = net_fxx(params, lower_bound, upper_bound)
    u_t = vmap(u_tf, (0, 0))(x, t)
    u_x = vmap(u_xf, (0, 0))(x, t)
    u_xx = vmap(u_xxf, (0, 0))(x, t)
    res = u_t + u.flatten() * u_x - nu * u_xx
    loss_f = jnp.mean((res.flatten()) ** 2)
    return loss_f


@jit
def predict_u(params, x_star, t_star, lower_bound, upper_bound):
    u_pred = vmap(net_u, (None, 0, 0, None, None))(
        params, x_star, t_star, lower_bound, upper_bound
    )
    return u_pred


nu = 0.01 / np.pi
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

(
    X,
    T,
    Exact,
    X_star,
    u_star,
    lower_bound,
    upper_bound,
    u_train,
    x_d,
    t_d,
    x_f,
    t_f,
    x_star,
    t_star,
) = BurgersPreprocessing()

params = init_network_params(layers, random.PRNGKey(1234))

opt_init, opt_update, get_params = optimizers.adam(5e-4)
opt_state = opt_init(params)
nIter = 20000 + 1
ld_list = []
lf_list = []


def loss_fn(params, x_f, t_f, x_d, t_d, lower_bound, upper_bound, nu, y_d):
    loss_res = 0.01 * loss_f(params, x_f, t_f, lower_bound, upper_bound, nu)
    data_loss = loss_data(params, x_d, t_d, lower_bound, upper_bound, y_d)
    return loss_res + data_loss


# @partial(jit, static_argnums=(0,))
@jit
def step(istep, opt_state, t_d, x_d, y_d, t_f, x_f, lower_bound, upper_bound):
    param = get_params(opt_state)
    g = grad(loss_fn, argnums=0)(
        param, x_f, t_f, x_d, t_d, lower_bound, upper_bound, nu, y_d
    )
    return opt_update(istep, g, opt_state)


pbar = trange(nIter)

for it in pbar:
    opt_state = step(
        it, opt_state, t_d, x_d, u_train, t_f, x_f, lower_bound, upper_bound
    )
    if it % 1 == 0:
        params = get_params(opt_state)
        l_d = loss_data(params, x_d, t_d, lower_bound, upper_bound, u_train)
        l_f = loss_f(params, x_f, t_f, lower_bound, upper_bound, nu)
        pbar.set_postfix({"Loss": l_d, "loss_physics": l_f})
        ld_list.append(l_d)
        lf_list.append(l_f)

l_list = []

u_pred = predict_u(params, x_star, t_star, lower_bound, upper_bound)
print(f"u_pred Shape: {u_pred.shape}")

error_u = jnp.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print("Error u: %e" % (error_u))
np.save("ld_list.npy", np.array(ld_list), allow_pickle=True)
np.save("lf_list.npy", np.array(lf_list), allow_pickle=True)


U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cupper_boundic")
Error = np.abs(Exact - U_pred)
