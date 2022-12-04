import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import optimizers
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# self adaptive pinn
class LambdaAdaptPINN: 

    def __init__(self, x_lb, x_ub, x0, u0, x, m, n, key, scale, layers, nu): 
        """
        """

        self.x_lb = x_lb  # lower boundary conditions
        self.x_ub = x_ub  # upper boundary conditions
        self.x0 = x0 # initial conditions
        self.u0 = u0 # initial conditions 
        self.x = x # collocation points in the domain (used to compute loss)

        N0 = self.x0.shape[0]
        NF = self.x.shape[0]

        # defining weights, biases, and lambdas 
        self.weights = self.initialise_NN_params(m, n, key, scale)[0]
        self.biases = self.initialise_NN_params(m, n, key, scale)[1]
        self.lambda_f = jnp.reshape(np.repeat(100, NF), (NF, -1))
        self.lambda_b = random.uniform(key, shape=[N0, 1])

        # define optimizer, layers, hyperparameters, and loss lists
        self.optimizer = optimizers.adam(0.001)
        self.layer_sizes = layers
        self.params = self.initialise_NN_params(self.layer_sizes, random.PRNGKey(0))
        self.nu = nu
        self.lb_list = []
        self.lf_list = []
    

    def initialise_NN_params(self, m, n, key, scale): 
        """
        A helper function to randomly initialize weights and biases for a 
        dense neural network layer.

        Args:
            m (int): _description_
            n (int): _description_
            key (_type_): _description_
            scale (_type_): _description_

        Returns:
            _type_: _description_
        """
        w_key, b_key = random.split(key)
        # might want to initialize with glorot?
        return scale * random.normal(w_key, (m, n)), jnp.zeros(n)

    @jit
    def predict(self, params, X):
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
            activations = tanh(jnp.matmul(activations, w) + b) 
            
        final_w, final_b = params[-1]
        logits = jnp.sum(jnp.matmul(activations, final_w) + final_b)
        return logits 

    @jit
    def net_u(self, params, X):
        """ """
        X = jnp.array([X])
        u = self.predict(params, X) 
        return u 

    @jit
    def net_ux(self, params):
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

    @jit
    def net_uxx(self, params):
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
    def func(self, X): 
        """ """
        return jnp.exp(X)

    @jit
    def loss_f(self, params, X, nu, lambda_f):
        """ """
        u = vmap(self.net_u, (None, 0))(params, X) 
        u_xxf = self.net_uxx(params)
        u_xx = vmap(u_xxf, (0))(X)
        fx = vmap(self.func, (0))(X)
        res = nu * u_xx - u - fx
        loss_f = jnp.mean((res.flatten()) ** 2)
        return lambda_f*loss_f

    @jit 
    def loss_b(self, params, lambda_b):
        """ """
        loss_b = (self.net_u(params, -1)-1) ** 2 + (self.net_u(params, 1)) ** 2
        return lambda_b*loss_b

    @jit
    def loss(self, params, X, nu, lambda_b, lambda_f):
        """ """
        lossf = self.loss_f(params, X, nu, lambda_f)
        lossb = self.loss_b(params, lambda_b)
        return lossb + lossf 

    @jit
    def optimise_net(self, istep, opt_state, X):
        """ """
        param = self.optimizer[-1](opt_state)
        g = grad(self.loss, argnums=0)(param, X, self.nu)
        return self.optimizer[1](istep, g, opt_state) 
    
    def optimize_lambda(): 
        pass 

    def train(self, nIter, X): 
        """ """
        pbar = trange(nIter)

        for it in pbar:
            opt_state = self.optimise_net(it, opt_state, X)

            if it % 1 == 0:
                params = self.optimizer[-1](opt_state)
                l_b = self.loss_b(params)
                l_f = self.loss_f(params, X, self.nu)
                l_b = self.loss_b(self.lambda_b)
                l_f = self.loss_f(params, X, self.nu)
                pbar.set_postfix({"Loss_res": l_f, "loss_bound": l_b})
                self.lb_list += l_b
                self.lf_list += l_f

        u_pred = vmap(self.predict, (None, 0))(params, X)

        return u_pred

    def callback(): 
        pass 

    def evaluate(): 
        pass 

    pass 

# evaluating, visualising, and plotting 

# random sampling
def random_residuals(): 
    pass 