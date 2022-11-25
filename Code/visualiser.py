import sys
sys.path.insert(0, 'Utilities/')
# from plotting import newfig, savefig
import matplotlib as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import jax
import time

from burgers_preprocessing import BurgersPreprocessing

class Plot():
    def __init__(self):
        pass
    def call(self, x, t, U_pred, Exact, u_train, X_u_train):
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