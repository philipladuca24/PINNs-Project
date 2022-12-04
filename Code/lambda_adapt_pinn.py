import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries.optimizers import adam, OptimizerState
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from jax.scipy.optimize import minimize
from jaxlib.xla_extension import DeviceArray
from functools import partial

# self adaptive pinn
class LambdaAdaptPINN: 

    def __init__(self, x_lb: int, x_ub: int, x: DeviceArray, layers: list[int], nu: float): 
        """
        """

        self.x_lb = x_lb  # lower boundary conditions
        self.x_ub = x_ub  # upper boundary conditions
        # self.x0 = x0 # initial conditions
        # self.u0 = u0 # initial conditions 
        self.x = x # collocation points in the domain (used to compute loss)

        # N0 = self.x0.shape[0]
        NF = self.x.shape[0]

        # defining weights, biases, and lambdas 
        self.lambda_f = jnp.array(jnp.reshape(np.repeat(100, NF), (NF, -1)))
        self.lambda_b = random.uniform(random.PRNGKey(0), shape=[NF, 1])

        # define optimizer, layers, hyperparameters, and loss lists
        self.opt_init, self.opt_update, self.get_params = adam(0.001)
        self.layer_sizes = layers
        self.params = self.init_network_params(self.layer_sizes, random.PRNGKey(0))
        self.nu = nu
        self.lb_list = []
        self.lf_list = []
    
    def random_layer_params(self, m: int, n: int, key: DeviceArray, scale: DeviceArray) -> DeviceArray:
        """
        A helper function to randomly initialize weights and biases for a 
        dense neural network layer.

        Args:
            m (int): _description_
            n (int): _description_
            key (DeviceArray): _description_
            scale (DeviceArray): _description_

        Returns:
            _type_: _description_
        """
        w_key, b_key = random.split(key)
        # might want to initialize with glorot?
        return scale * random.normal(w_key, (m, n)), jnp.zeros(n)

    def init_network_params(self, sizes: list[int], key: DeviceArray) -> DeviceArray:
        """
        Initialize all layers for a fully-connected neural network with 
        sizes "sizes".

        Args:
            sizes (list[int]): _description_
            key (DeviceArray): _description_

        Returns:
            DeviceArray: _description_
        """
        keys = random.split(key, len(sizes))
        # why is this scaling used?
        return [
            self.random_layer_params(m, n, k, 2.0 / (jnp.sqrt(m + n)))
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]

    @jit
    def predict(self, params: list, X: DeviceArray) -> DeviceArray:
        """
        Per example predictions

        Args:
            params (list): _description_
            X (DeviceArray): _description_

        Returns:
            DeviceArray: _description_
        """
        activations = X

        for w, b in params[:-1]:
            activations = tanh(jnp.matmul(activations, w) + b) 
            
        final_w, final_b = params[-1]
        return jnp.sum(jnp.matmul(activations, final_w) + final_b)         

    @jit
    def net_u(self, params: list, X: DeviceArray) -> DeviceArray:
        """ """
        X = jnp.array([X])
        return self.predict(params, X)  

    
    def net_ux(self, params: list) -> DeviceArray:
        """
        Define neural network for first spatial
        derivative of u(x): u'(x).

        Args:
            params (_type_): _description_
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        def ux(X):
            return grad(self.net_u, argnums=1)(params, X) 
        
        return jit(ux)

    
    def net_uxx(self, params: list) -> DeviceArray:
        """
        Define neural network for second spatial
        derivative of u(x): u''(x).

        Args:
            params (_type_): _description_
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        def uxx(X):
            u_x = self.net_ux(params)
            return grad(u_x)(X) 
        
        return jit(uxx)

    @jit
    def funx(self, X: DeviceArray) -> DeviceArray: 
        """ """
        return jnp.exp(X)

    @jit
    def loss_f(self, params: list, X: DeviceArray, nu: float, lambda_f: DeviceArray) -> DeviceArray:
        """ """
        u = vmap(self.net_u, (None, 0))(params, X) 
        u_xxf = self.net_uxx(params)
        u_xx = vmap(u_xxf, (0))(X)
        fx = vmap(self.funx, (0))(X)
        res = nu * u_xx - u - fx
        loss_f = lambda_f * jnp.mean((res.flatten()) ** 2)
        return loss_f

    @jit 
    def loss_b(self, params: list, lambda_b: DeviceArray) -> DeviceArray:
        """ """
        loss_b = lambda_b * ((self.net_u(params, -1)-1) ** 2 + (self.net_u(params, 1)) ** 2)
        return loss_b

    @jit
    def loss(self, params: list, X: DeviceArray, nu: float, lambda_b: DeviceArray, lambda_f: DeviceArray) -> DeviceArray:
        """ """
        lossf = self.loss_f(params, X, nu, lambda_f)
        lossb = self.loss_b(params, lambda_b)
        return lossb + lossf 

    @partial(jit, static_argnums=(0,))
    def optimise_net(self, istep: int, opt_state: OptimizerState, X: DeviceArray, lambda_b: DeviceArray, lambda_f: DeviceArray) -> DeviceArray:
        """ """
        param = self.get_params(opt_state)
        print(type(param), "param")
        print(type(X), "X")
        print(type(self.nu), "nu")
        print(type(lambda_b), "lambda_b")
        print(type(lambda_f), "lambda_f")
        g = grad(self.loss, argnums=0)(param, X, self.nu, lambda_b, lambda_f)
        opt_state = self.opt_update(istep, g, opt_state) 
        print(type(opt_state), "opt_state")
        return opt_state
    
    @jit
    def optimise_lambda(self, params: list, X: DeviceArray, nu: float, lambda_b: DeviceArray, lambda_f: DeviceArray): 
        lamb = minimize(self.loss_b, lambda_b, args=[params, lambda_b], method="L-BFGS-B")
        lamf = minimize(self.loss_f, lambda_f, args=[params, X, nu, lambda_f], method="L-BFGS-B")
        return lamb, lamf

    def train(self, nIter: int, X: DeviceArray): 
        """ """
        pbar = trange(nIter)
        opt_state = self.opt_init(self.params)
        lambda_b = self.lambda_b
        lambda_f = self.lambda_f

        for it in pbar:
            opt_state = self.optimise_net(it, opt_state, X, lambda_b, lambda_f)
            if it % 1 == 0:
                opt_params_net = self.get_params(opt_state)
                opt_state_lambda = self.optimise_lambda(opt_params_net, X, self.nu, self.lambda_b, self.lambda_f)

                self.lambda_b = opt_state_lambda[0]
                self.lambda_f = opt_state_lambda[1]


                l_b = self.loss_b(opt_params_net, self.lambda_b)
                l_f = self.loss_f(opt_params_net, X, self.nu, self.lambda_f)
                pbar.set_postfix({"Loss_res": l_f, "loss_bound": l_b})
                self.lb_list += l_b
                self.lf_list += l_f

        u_pred = vmap(self.predict, (None, 0))(opt_params_net, X)

        return u_pred, self.lb_list, self.lf_list

    def callback(): 
        pass 

    def evaluate(): 
        pass 

if __name__ == "__main__":
    nu = 10 ** (-3)
    layer_sizes = [1, 20, 20, 20, 1]
    nIter = 20000 + 1
    x = jnp.arange(-1, 1.1, 0.1)

    model = LambdaAdaptPINN(-1, 1, x, layer_sizes, nu)
    
    u_pred, lb_list, lf_list = model.train(nIter, x)
    
    plt.plot(x, u_pred)
    plt.plot(nIter, lb_list)
    plt.plot(nIter, lf_list)
    plt.show()
    

# evaluating, visualising, and plotting 
