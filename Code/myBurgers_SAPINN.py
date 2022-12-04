# -*- coding: utf-8 -*-
"""
Created on 
@author: Somdatta Goswami
"""
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import scipy.io
import time
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting import newfig
from scipy.interpolate import griddata

#%%
def LHSample(D, bounds, N):
    """
    A function used to sample random points in a domain
    # :param D: Number of parameters
    # :param bounds:  [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
    # :param N: Number of samples
    # :return: Samples
    """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # Stretching the sampling
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds,
           out=result)
    return result


#%%
""" PINNs model """
class SAPINN:
    # Initialize the class
    def __init__(self, x_lb, t_lb,
                       x_ub, t_ub,
                       x0, t0, u0,
                       x_f, t_f,
                       layers):
        
        N_f = x_f.shape[0]
        N_0 = x0.shape[0]
        
        self.x_lb = x_lb  # boundary conditions
        self.t_lb = t_lb
        self.x_ub = x_ub  # boundary conditions
        self.t_ub = t_ub
        self.x0 = x0  # boundary conditions
        self.t0 = t0
        self.u0 = u0
        self.x_f = x_f  # points inside the domain (used to compute the PDE loss)
        self.t_f = t_f
                
        # layers
        self.layers = layers
        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.lambda_f = tf.Variable(tf.reshape(tf.repeat(100.0, N_f),(N_f, -1)))
        self.lambda_u0 = tf.Variable(tf.random.uniform([N_0, 1]))

        # tf placeholders and graph
        # placeholders are used to receive network inputs
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.saver = tf.train.Saver()
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, 1]) 
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, 1]) 
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1]) 
        
        # physics informed neural networks
        # forward computation for boundary conditions
        self.u_lb_pred = self.net_u(self.x_lb_tf, self.t_lb_tf) 
        self.u_ub_pred = self.net_u(self.x_ub_tf, self.t_ub_tf) 
        self.u0_pred = self.net_u(self.x0_tf, self.t0_tf) 
        # compute residuals 
        self.f_pred = self.net_equation(self.x_f_tf, self.t_f_tf)  
        
        # loss
        self.loss_log = []
        self.loss_b = tf.reduce_mean(tf.square(self.u_lb_pred)) + \
                        tf.reduce_mean(tf.square(self.u_ub_pred)) 
        self.loss_0 = tf.reduce_mean(tf.square(self.lambda_u0*( self.u0_tf - self.u0_pred) )  )
        self.loss_e = tf.reduce_mean(tf.square(self.lambda_f*self.f_pred)) 
        
        self.loss = self.loss_b + self.loss_e + self.loss_0
                
        # ADAM optimizers
        self.optimizer = tf.train.AdamOptimizer(0.005)
        # self.grads = self.optimizer.compute_gradients(self.loss)
        # self.train_op = self.optimizer.apply_gradients(self.grads)
        # self.train_op = self.optimizer.minimize(self.loss)
        self.grads_weights = self.optimizer.compute_gradients(self.loss, self.weights)
        self.grads_biases = self.optimizer.compute_gradients(self.loss, self.biases)
        self.grads_lambda_u0 = self.optimizer.compute_gradients(self.loss, self.lambda_u0)
        self.grads_lambda_f = self.optimizer.compute_gradients(self.loss, self.lambda_f)
    
        self.grads_lambda_u0_minus = [(-gv[0], gv[1]) for gv in self.grads_lambda_u0]
        self.grads_lambda_f_minus = [(-gv[0], gv[1]) for gv in self.grads_lambda_f]
    
        self.op_w = self.optimizer.apply_gradients(self.grads_weights)
        self.op_b = self.optimizer.apply_gradients(self.grads_biases)
        self.op_lamU0 = self.optimizer.apply_gradients(self.grads_lambda_u0_minus)
        self.op_lamF = self.optimizer.apply_gradients(self.grads_lambda_f_minus)
        
        
        
        
        # lbfgs optimizers
        self.lambda_f_const = tf.placeholder(tf.float32, shape=[None, 1])
        self.lambda_u0_const = tf.placeholder(tf.float32, shape=[None, 1]) 
        self.loss_0_const = tf.reduce_mean(tf.square(self.lambda_u0_const*( self.u0_tf - self.u0_pred) )  )
        self.loss_e_const = tf.reduce_mean(tf.square(self.lambda_f_const*self.f_pred)) 
        self.loss_lbfgs = self.loss_b + self.loss_e_const + self.loss_0_const
        self.optimizer_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(self.loss_lbfgs, 
                                        method = 'L-BFGS-B', 
                                        options = {'maxiter': 30000,
                                                   'maxfun': 30000,
                                                   'maxcor': 50,
                                                   'maxls': 50,
                                                   'ftol' : 1.0 * np.finfo(float).eps})
    

        # whole model initialization 
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # ================================================    
    #   network initialization
    # ================================================    
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    # ================================================    
    #   computational graph - forward computation
    # ================================================
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    # ================================================    
    #   from input to output - calling the graph
    # ================================================
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    # ================================================    
    #   from input to residuals 
    # ================================================
    def net_equation(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        u_x = tf.gradients(u,x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_t = tf.gradients(u,t)[0]
        f = u_t + u*u_x - (0.01/tf.constant(math.pi))*u_xx
        return f
    
    # ================================================    
    #   ADAM
    # ================================================
    def train(self, num_epochs, batch_size):
        start_time = time.time()
        for epoch in range(num_epochs):
            N_f = self.x_f.shape[0]
            for it in range(0, N_f, batch_size):

                tf_dict = {self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                           self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                           self.x0_tf: self.x0, self.t0_tf: self.t0, self.u0_tf: self.u0,
                           self.x_f_tf: self.x_f, 
                           self.t_f_tf: self.t_f}
                
                self.sess.run([self.op_w,self.op_b], tf_dict)
                self.sess.run([self.op_lamF,self.op_lamU0], tf_dict)
            
                # Print
                if epoch % (10) == 0:
                    (loss_value, loss_b,
                     loss_e,loss_0) = self.sess.run([self.loss, self.loss_b, 
                                                     self.loss_e, self.loss_0 ],tf_dict)
    
                    elapsed = time.time() - start_time
                    print('---------------------------------------------------')
                    print('Epoch: %d, Time: %.2f'
                          %(epoch, elapsed))
                    print('Epoch: %d, loss: %.3e, loss_b: %.3e, loss_e: %.3e, loss_0: %.3e'
                          %(epoch, loss_value, loss_b, loss_e, loss_0))
                    start_time = time.time()
                    
                    self.loss_log.append([loss_value, loss_b, loss_e, loss_0])
    
    # ================================================    
    #   L-BFGS-B
    # ================================================
    def callback(self, loss_value, loss_b, loss_e, loss_0):
        print('loss: %.3e, loss_b: %.3e, loss_e: %.3e, loss_0: %.3e'
                          %(loss_value, loss_b, loss_e, loss_0))
        self.loss_log.append([loss_value, loss_b, loss_e, loss_0])
    def train_lbfgs(self, lam_u0, lam_f):
        tf_dict = {self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                    self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                    self.x0_tf: self.x0, self.t0_tf: self.t0, self.u0_tf: self.u0,
                    self.x_f_tf: self.x_f, self.t_f_tf: self.t_f ,
                    self.lambda_u0_const:lam_u0, self.lambda_f_const:lam_f}
        self.optimizer_lbfgs.minimize(self.sess, feed_dict=tf_dict,
                                      fetches=[self.loss_lbfgs, 
                                                self.loss_b, 
                                                self.loss_e_const, 
                                                self.loss_0_const],
                                      loss_callback=self.callback)    
  
    # ================================================    
    #   making prediction - from input to output 
    # ================================================
    def predict(self, x_star, t_star):
        tf_dict = {self.x0_tf: x_star, self.t0_tf: t_star}
        u_star = self.sess.run(self.u0_pred, tf_dict)
        tf_dict = {self.x_f_tf: x_star, self.t_f_tf: t_star}
        f_star = self.sess.run(self.f_pred, tf_dict)
        return u_star, f_star
    

    
    #%%
######################################################################
######################## main funtion ################################
######################################################################
if __name__ == "__main__": 
    
    
    # network architecture 
    # [num of input] + num_layer*[num_node] + [num of output]   
    # input: x,y
    # output: u,v,p
    num_layer = 8
    num_node = 20
    layers = [2] + num_layer*[num_node] + [1]   
    
    lb = np.array([-1.0]) #x upper boundary
    ub = np.array([1.0]) #x lower boundary
    
    N0 = 100
    N_b = 25 #25 per upper and lower boundary, so 50 total
    N_f = 10000  
    
    
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
    #load data, from Raissi et. al
    data = scipy.io.loadmat('burgers_shock.mat')
    
    t = data['t'].flatten()[:,None].astype(np.float32)
    x = data['x'].flatten()[:,None].astype(np.float32)
    Exact = data['usol'].astype(np.float32)
    Exact_u = np.real(Exact)
    
    #grab random points off the initial condition
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    t0 = np.zeros_like(x0)
    u0 = Exact_u[idx_x,0:1]
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    x_lb = np.zeros_like(tb) + lb[0]
    t_lb = tb
    x_ub = np.zeros_like(tb) + ub[0]
    t_ub = tb
    
    # randomly choose residual points
    xye = LHSample(2, [[lb[0],ub[0]], [0,1]], N_f)
    x_f = xye[:, 0:1]
    t_f = xye[:, 1:2]

    

    ################ Training   #################################
    #############################################################
    # construct model
    model = SAPINN(x_lb, t_lb, x_ub, t_ub, x0, t0, u0, x_f, t_f, layers)

    time_startTraining = time.time()
    # MODEL TRAINING - 10000 iterations
    model.train(num_epochs = 10000, batch_size = N_f)
    lam_u0 = model.sess.run(model.lambda_u0)
    lam_f = model.sess.run(model.lambda_f)
    model.train_lbfgs(lam_u0, lam_f)
    elapsed_time = time.time() - time_startTraining
    # load the history of the training loss function
    loss_log = model.loss_log
    
    #generate mesh to find U0-pred for the whole domain
    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.T.flatten()[:,None]
    
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1])
    
    # Get preds
    u_pred, f_u_pred = model.predict(X_star[:,0:1], X_star[:,1:2])
    #find L2 error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    print('Training time: %e' % (elapsed_time))
    print('Error u: %e' % (error_u))
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])
    
    fig, ax = newfig(1.3, 1.0)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_u[:,25], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$' % (t[25]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_u[:,50], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = %.2f$' % (t[50]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_u[:,75], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    
    #show u_pred across domain
    fig, ax = plt.subplots()
    
    ec = plt.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                extent=[0.0, 1.0, -1.0, 1.0],
                origin='lower', aspect='auto')
    
    ax.autoscale_view()
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    cbar = plt.colorbar(ec)
    cbar.set_label('$u(x,t)$')
    plt.title("Predicted $u(x,t)$",fontdict = {'fontsize': 14})
    plt.show()
    
    # Show F_U_pred across domain, should be close to 0
    fig, ax = plt.subplots()
    
    ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
                extent=[0.0, math.pi/2, -5.0, 5.0],
                origin='lower', aspect='auto')
    
    ax.autoscale_view()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    cbar = plt.colorbar(ec)
    cbar.set_label('$\overline{f}_u$ prediction')
    plt.show()
    
    # collocation point weights
    plt.scatter(t_f, x_f, c = model.sess.run(model.lambda_f), s = model.sess.run(model.lambda_f)/5)
    plt.show()
    
            
        
            
            
            
            
            
            