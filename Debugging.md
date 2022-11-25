## Debugging:

# 24/11/22 - Visualiser -> newfig / plotting / plt.figure() error:
"Nov 24 - code is training, pre turning code into classes" push:

- TypeError: error message below. Error probably coming from plt.figure(1.0, 1.1) on line 236.
Likely stemming from latent error of "ImportError: cannot import name 'newfig' from 'plotting'". 
Tried to solve using: https://github.com/maziarraissi/PINNs/issues/36
Ultimately 'solved' by commented out the import line (line 17) and replaced it with line 18. 
Newfig doesn't appear in the matplotlib documentation so thought it was referring to plt.figure 
instead. Note: plotting seems to be an import from matplotlib.pyplot but not extremely clear either.
"
Traceback (most recent call last):
    File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/pinn_bugers_jax.py", line 201, in <module>
    File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/_api/deprecation.py", line 454, in wrapper
        return func(*args, **kwargs)
    File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/pyplot.py", line 783, in figure
        manager = new_figure_manager(
    File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/pyplot.py", line 359, in new_figure_manager
        return _get_backend_mod().new_figure_manager(*args, **kwargs)
    File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/backend_bases.py", line 3504, in new_figure_manager
        fig = fig_cls(*args, **kwargs)
    File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/_api/deprecation.py", line 454, in wrapper
        return func(*args, **kwargs)
    File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/matplotlib/figure.py", line 2473, in __init__
        self.bbox_inches = Bbox.from_bounds(0, 0, *figsize)
TypeError: Value after * must be an iterable, not float
"

- TypeError: error message below. Error is a continuation of the error above.
"
    Traceback (most recent call last):
        File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/pinn_bugers_jax.py", line 241, in <module>
    TypeError: cannot unpack non-iterable Figure object
"
